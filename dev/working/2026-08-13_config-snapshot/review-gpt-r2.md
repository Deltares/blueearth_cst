## Verdict
verdict: revise
doc_version: config-snapshot-design-v2.md

## Findings
### ext2-1  [blocking]
- section: 5.7 The run journal
- finding: `snapshot_config` cannot produce the specified one-line-per-run journal or reliable revision stamp because it remains an ordinary terminal Snakemake rule with stable outputs. When its config inputs and outputs are unchanged, Snakemake skips it; a same-config invocation or a checkout change affecting another workflow module therefore creates no journal entry and may leave `run_record.yml` stamped with the previous commit.
- rationale: A direct Snakemake run can execute changed computational code while reporting the old toolbox revision, and successful, failed, and no-op invocations can be absent from the journal. This defeats R5, R7, and the claim that the journal identifies invocations.
- suggested_fix: Move journal emission to workflow lifecycle hooks that execute for every non-dry invocation, append one terminal record with a unique invocation ID and status, and embed that ID in terminal provenance. Separately make toolbox identity an explicit tracked input or parameter of `snapshot_config` so a revision change refreshes `run_record.yml`.

### ext2-2  [major]
- section: 5.4 Two digests with explicit, different meanings
- finding: Section 5.7 embeds `effective_config_sha256` in terminal outputs even though §5.4 deliberately excludes referenced-input contents and, unlike the current implementation, excludes `advanced_settings`. Consequently it is the wrong digest for the proposed staleness check.
- rationale: Changing a custom template, catalog, committed advanced setting, or toolbox revision can change `run_inputs_sha256` while leaving `effective_config_sha256` unchanged. Old outputs will then match the new `run_record.yml` on the only embedded field and appear current. The statement that this digest is “unchanged in meaning from today” is also false: `provenance.py:123-148` currently includes advanced settings.
- suggested_fix: Embed and compare `run_inputs_sha256` in terminal provenance. Define whether consumed advanced settings belong in `effective_config_sha256`, but include them directly in at least the run-input digest and its mutation tests rather than relying only on the commit changing.

### ext2-3  [major]
- section: 5.6 The values-used record (R3, R6)
- finding: The proposed carrier still does not completely specify the actual values handed to HydroMT. `build_wflow_model.py:237-268` removes configured arguments and injects derived objects and `lulc_mapping_fn`, so serializing `read_parameter_steps` output records the input template rather than the normalized call values. The promised equivalent waterbodies record is also absent from §5.8’s layout and §6’s implementation inventory.
- rationale: An analyst cannot reconstruct why HydroMT received a derived mapping or distinguish template values from adapter substitutions. Waterbody setup values remain unrecorded after their template copy is removed. Thus the accepted R3 fix is incomplete.
- suggested_fix: Define a serializable normalized values-used schema written after adapter normalization, representing injected datasets by stable source references. Name and declare a corresponding output for rule 1.08, and add both records to the layout, implementation inventory, and tests.

### ext2-4  [major]
- section: 5.4 Two digests with explicit, different meanings
- finding: `run_inputs_sha256` is described as answering whether two runs “saw the same inputs,” but it covers configuration projections, toolbox identity, and selected referenced-file bytes—not generated inputs or the scientific data addressed by catalogs.
- rationale: Two runs can receive different remote or mutable dataset contents through an unchanged catalog and still have the same `run_inputs_sha256`. The broad name and equivalence claim invite this digest to be cited as scientific run identity when it is only configuration-input identity.
- suggested_fix: Rename it to `run_configuration_inputs_sha256` and explicitly exclude scientific data identity, or extend the contract to incorporate the existing resolved-source provenance and generated-input identities.

### ext2-5  [minor]
- section: 5.3 Scope by consumed keys, declared and tested — not by section ownership
- finding: The mandatory mutation test proves only that declared keys affect the digest; it cannot detect an actual config read omitted from the declaration—the exact failure mode behind `gpt-1`.
- rationale: A future cross-section read can change outputs while the digest and mandated tests remain unchanged.
- suggested_fix: Add a completeness check for cross-section accesses, or route config reads through a projection-aware accessor whose observed paths are compared with the declaration.

### ext2-6  [minor]
- section: 5.2 One consolidated `run_record.yml` per workflow
- finding: `toolbox.version` has no defined source, and the repository has no declared project/package version in `pyproject.toml` or a discoverable `__version__`.
- rationale: Implementers must invent a value, omit the field, or fail while resolving package metadata, producing inconsistent records.
- suggested_fix: Remove the field or define its exact derivation and unavailable-value behavior.

## Regression check
- `gpt-1` — resolved-with-new-defect: WF3’s current cross-section reads are included, but the mandated test does not establish declaration completeness.
- `gpt-2` — not resolved: the values-used carrier omits adapter-normalized values, and the revision stamp is not guaranteed to refresh.
- `gpt-3` — resolved: the per-file recoverability predicate covers custom paths outside the toolbox.
- `gpt-4` — resolved-with-new-defect: referenced-file hashes now exist in a second digest, but terminal staleness checks embed the narrower digest.
- `gpt-5` — resolved: role-stable destinations and collision refusal remove basename overwrites.
- `fbl-1` — not resolved: a named carrier exists, but it does not yet record all values actually handed to HydroMT.
- `fbl-2` — resolved: the design includes report-only cleanup and orders fixture cleanup before inventory changes.
- `fbl-3` — resolved-with-new-defect: revision metadata is consolidated, but commit-only changes need not rerun its writer.
- `fbl-4` — not resolved: the journal is not invocation-driven, lacks output correlation/status, and the terminal marker misses relevant input changes.
- `fbl-5` — resolved: recoverability is decided per file rather than per bin.
- `fbl-6` — resolved: the schema version is bumped to 2.
- `fbl-7` — resolved: the verified test, README, and dead-helper updates are included in §6.
- `fbl-8` — resolved: §5.8 correctly identifies all three baseline-fingerprinted flat copies.
- `fbl-9` — resolved-with-new-defect: WF1 and WF3 gain digest readers, but the chosen digest can falsely certify stale outputs.