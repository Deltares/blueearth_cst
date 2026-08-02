# M2 design decisions

Decisions taken during M2 that affect long-term shape of the repo.
Locked here so future-you remembers *why*, not just *what*.

## Tooling: hybrid pixi + native Julia (not full pixi)

**Decision.** Pixi manages Python and R. Julia/Wflow.jl is managed
via native `Project.toml` + `Manifest.toml`. A `pixi run install`
task wraps both for the single-command install story.

**Why.** The roadmap's stated risk for full pixi was R coverage
(weathergenr in particular); Julia under pixi is a separate uncertainty.
Going hybrid removes one variable: native Julia tooling is mature,
lockfile-based, used by Wflow.jl's own docs, and produces a
`Manifest.toml` with full transitive-dep pinning. Pixi's `[tasks]`
layer can wrap the Julia install step so the user-facing install
story stays one command.

**Reversible.** If full-pixi becomes worth it later, add `julia` to
`pixi.toml`'s `[dependencies]`, point the install task at the
pixi-managed julia, and delete the standalone Julia install step from
the README. Project.toml stays where it is.

## weathergenr: pin via pixi task, not vendor

**Decision.** weathergenr is installed at env-setup time via
`devtools::install_github("tanerumit/weathergenr", ref="v1.2.0",
upgrade="never")`, invoked by a pixi task (`pixi run install-rdeps`).
**Not** vendored under `vendor/weathergenr/`.

**Why.** Smaller repo footprint; keeps the weathergenr fork as a
first-class R package rather than a snapshot under `vendor/`. The
trade-off is an internet dependency on install (fine — install is
infrequent) and that local mid-flight fixes to weathergenr require a
push to the fork plus a tag bump. The user already maintains the
fork at `tanerumit/weathergenr` so the fork-bump cost is low.

**Trigger to revisit.** If the install starts failing because of
GitHub availability, devtools API drift, or compat issues with newer
R toolchains, vendor as a fallback.

## Julia version pinned to 1.12 (not Dockerfile's 1.8.2)

**Decision.** Project.toml `[compat] julia = "1.12"`. The Dockerfile
still says `julia_version=1.8.2`; that gets updated when Linux/Docker
work resumes (deferred per the roadmap).

**Why.** 1.12.6 is what produced the M1 baseline locally and what
Wflow.jl 0.8.1 currently supports cleanly. Sticking to 1.8.2 would
require pinning Wflow.jl to whatever version was current at the
Dockerfile's build time — also unpinned. Both stale; better to pick
the working pair.
