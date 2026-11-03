# Frozen GEV L-moment controls

These fixtures provide independent parity and high-precision numerical controls for
`tests/test_gev_lmoments.py` and `tests/test_indicator_registry.py`. They are frozen test inputs,
not production implementations. Missing controls must fail the tests rather than skip them.

Copied without byte changes from private historical evidence at source revision
`7c784c0c1123125780ccd811b308c7dabcf44f7c`. Original evidence remains unchanged in the private
checkout. The following source paths are provenance citations, not public checkout prerequisites:

- `dev/records/milestones/r12/implementation/evidence/gf15-lmoments-readiness/candidate_adapter.py`
- `dev/records/milestones/r12/implementation/evidence/gf15-lmoments-readiness/results/attempt-2/numerical-controls.json`

Git blob identities (independent of checkout newline conversion):

| Fixture | Source blob |
|---|---|
| `candidate_adapter.py` | `8e1f21b71e864bedd9bd76f4c8c49c842b656e7e` |
| `numerical-controls.json` | `77cdecf59283c27a2c05ea5af5af4722a168f58b` |

The candidate is the implementation used for the accepted GF15 8x qualification. The attempt-2
controls retain independently computed 80–120 decimal digit references and expected float64 bit
patterns. Preserve both files unchanged; do not regenerate expected values from production code.
