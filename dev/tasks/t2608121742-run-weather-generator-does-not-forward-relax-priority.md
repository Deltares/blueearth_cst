---
title: "`run_weather_generator` does not forward `relax_priority`"
type: watch-item
area: upstream / weathergenr
origin: WF3 weathergenr 1.2.0 alignment
created: 2026-08-12
updated: 2026-08-12
---

> [!note] Overview
> **What** — `weathergenr::run_weather_generator` forwards every `generate_weather` argument except `relax_priority`, so rule 3.06 cannot set it.
> **Why** — Not currently harmful: the argument keeps its upstream default. It is recorded so the config's omission reads as a known upstream gap rather than an oversight, and so the key is restored deliberately rather than rediscovered.
> **Trigger** — Upstream `tanerumit/weathergenr` forwards `relax_priority` from `run_weather_generator`, OR a run needs a non-default WARM bound-relaxation order.

## Detail

Rule 3.06 called `weathergenr::generate_weather` directly until 2026-08-12,
when it swapped to `run_weather_generator` — the wrapper that runs the same
generation and then the evaluation pass (`prepare_evaluation_data` +
`evaluate_weather_generator`), whose diagnostic plots are the reason for the
swap.

The wrapper takes the generation arguments as one `config` list. Measured
against weathergenr 1.2.0 by comparing `formals(generate_weather)` with the
call inside `body(run_weather_generator)`, it forwards **19 of 20** settable
arguments. The exception is `relax_priority` — the order in which the WARM
accept bounds (`wavelet`, `sd`, `tail_low`, `tail_high`, `mean`) are relaxed
when the sample pool underfills.

`config/defaults/weathergen_config.yml` therefore **omits** `relax_priority`,
and `interchange_contracts._WG3_GENERATE_WEATHER_KEYS` does not pin it. Both
carry a comment saying why. Setting it would produce a key that reads as a
live setting and reaches nothing — the defect C34 found in the retired
`evaluate.model`, and the reason WG-3 pins the key set at all.

## Options if the trigger fires

1. **Upstream forwards it** (preferred): add `relax_priority` back to the
   `generate_weather:` section, pin it in `_WG3_GENERATE_WEATHER_KEYS`, and
   delete the three explanatory comments. No other change — the section is
   already handed to the wrapper verbatim.
2. **A run needs a non-default order before upstream fixes it**: inline the
   wrapper's three calls in `generate_weather.R` (`generate_weather` →
   `prepare_evaluation_data` → `evaluate_weather_generator`, all exported).
   This keeps the evaluation plots and restores the argument, at the cost of
   owning ~15 lines that track an upstream function. This option was weighed
   at the time of the swap and declined in favour of staying on the maintained
   entry point.

## Refs

- `config/defaults/weathergen_config.yml` — `generate_weather:`, the
  `warm_filter_bounds` comment block states the omission.
- `blueearth_cst/shared/interchange_contracts.py` —
  `_WG3_GENERATE_WEATHER_KEYS`.
- `blueearth_cst/weathergen/generate_weather.R` — the `run_weather_generator`
  call.
- Related: [[t2608071225-weathergenr-write-netcdf-does-not-propag]],
  [[t2608071226-weathergenr-s-wavelet-minimum]] — the other two open
  upstream-weathergenr watch-items.
