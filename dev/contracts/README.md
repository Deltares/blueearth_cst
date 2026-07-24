# Interchange contracts (P3-2b)

The pipeline's two substitution seams, pinned as explicit, machine-checked
contracts (design: `dev/p32b/interchange-contracts-design.md`, ACCEPTED
2026-07-24):

- [`weather-generator-seam.md`](weather-generator-seam.md) — WG-1..WG-6: what
  a replacement stochastic weather generator consumes and must produce.
- [`hydrological-model-seam.md`](hydrological-model-seam.md) — HM-1..HM-7
  (HM-6a/6b): what a replacement hydrological model consumes and must
  produce.

Each doc carries the per-artifact contract table (pinned /
pinned-as-reliance / deliberately-unpinned), a bounded-substitution
walkthrough (the exact repo files a swap touches), and a validator index.

**Checking.** Validators live in
`blueearth_cst/shared/interchange_contracts.py` (pure `-> list[str]`
divergence reports; wired into no pipeline rule) and are exercised by
`tests/test_interchange_contracts.py`. Coverage follows the design's §5.5
counting axis verbatim: 15 validators / 30 always-run synthetic pass-fail
tests; with the `examples/test_local` fixture present, 12 integration checks
are green and 3 temp()-content cases are skip-until-captured (`--notemp`
procedure in each doc's validator index); on a fixtureless checkout all
integration cases skip under the named `_FIXTURE_ABSENT` reason — a visible
unmet-integration condition, never a silent green (`pytest -rs` shows the
split). chirps-branch facts are documented but not fixture-verified (era5
fixture only).
