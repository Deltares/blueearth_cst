# C34 — one recorded decision per unpassed weathergenr argument

Landed 2026-08-08 on `refactor/r11-p2-rename` (R11 P2, commit 4). Specification:
`dev/milestones/r09/wf3-change-requests.md` **CR-5b / C34**, with findings
**F13–F16**.

**The rule C34 states, and this document discharges:** *not* "expose
everything" — most upstream defaults are right, and surfacing them would bloat
every project config. Each unpassed argument gets **one recorded decision**:
*surfaced*, or *default accepted deliberately*, with a reason. An unexamined
default is not a choice.

## Provenance of the argument lists, and its limit

The 19 unpassed arguments below are **the register's**, read from the installed
`tanerumit/weathergenr@v1.2.0` when CR-5b was written (F13, F14). They were
**not** re-read here: `weathergenr` comes from `pixi run install`, and this
worktree has had only `pixi install` — `requireNamespace("weathergenr")` returns
FALSE. Verified rather than assumed, and stated because it bounds what this
document can claim: it records decisions against the register's signature
reading, not against a fresh one.

**The three surfaced arguments are therefore UNEXECUTED.** No R test harness
exists (ruled at R5: Python helpers only), and the R cannot run without the
package. `save_plots`, `seed` and `pet_method` first execute at rules 3.11 and
3.12 in a real WF3 run. Same standing caveat CR-5 recorded for C29: **run WF3
from the primary checkout before treating this as done.**

---

## `generate_weather()` — 5 unpassed of 24 (F13)

| argument | default | decision | reason |
| --- | --- | --- | --- |
| `save_plots` | `TRUE` | **SURFACED** as `generateWeatherSeries.save.plots` | It is the live control that `evaluate.model` used to be. v1.2.0 split evaluation into its own exports, so the config's `evaluate.model` reached **nothing** — a user setting it `FALSE` still got every plot. Surfacing restores the behaviour the key already claimed |
| `warm_filter_bounds` | `list()` | default accepted deliberately | New in 1.2.0: acceptance bounds on the generated annual series. An empty list means "no additional filter", which is the pre-1.2.0 behaviour every existing project was validated under. Changing it is a **scientific** choice needing its own evidence, not a plumbing default |
| `relax_priority` | `c("wavelet","sd","tail_low","tail_high","mean")` | default accepted deliberately, **flagged** | Which distributional criterion is sacrificed when the warm-pool filter cannot be met. Genuinely scientific, and currently decided by an upstream default. Only reachable once `warm_filter_bounds` is set, so it is inert today — but it is the argument most worth revisiting first |
| `n_cores` | `NULL` | default accepted deliberately | We already pass `parallel`; `NULL` lets weathergenr choose. A second parallelism knob belongs with `julia_threads` in `advanced_settings.yml`, which is unit D's surface, not this one |
| `verbose` | `FALSE` | default accepted deliberately | Diagnostic chatter. Our logging contract is `tee_to_log` + `log_row`; a second, upstream-formatted stream into the same log part would fight it |

## `apply_climate_perturbations()` — 14 unpassed of 25 (F14)

| argument | default | decision | reason |
| --- | --- | --- | --- |
| `seed` | `NULL` | **SURFACED** — the generator's own `seed` is now passed | Generation was seeded and the perturbation was not, so two halves of one experiment had different reproducibility guarantees and nobody chose that (F15). If the function is deterministic this is a no-op; if it is not, the run becomes reproducible. Either way the asymmetry stops being an oversight |
| `pet_method` | `"hargreaves"` | **SURFACED** as `generateWeatherSeries.pet.method`, at the upstream default | PET is computed **twice** in this chain by two different methods — here, and again from the perturbed temperature by rule 3.14 — and neither was chosen (F16). Surfacing states this step's method. Whether the first result is used at all is F16's open half and is **not** settled here |
| `precip_occurrence_factor` | `NULL` | **not surfaced — needs a stress dimension** | A real stress dimension (wet/dry day frequency). Adding it means a design-table column and a results column together, which C28's hard stop refuses by design. Belongs to the milestone that widens the design, not to a plumbing audit |
| `precip_occurrence_transient` | `TRUE` | not surfaced | Its transient flag; inert while the dimension is unreachable |
| `precip_intensity_threshold` | `0` | default accepted deliberately | Wet-day threshold. `0` means "any non-zero day is wet", which is what the current occurrence statistics already assume |
| `exaggerate_extremes` | `FALSE` | **not surfaced — needs a stress dimension** | A real stress dimension (extreme intensification). `FALSE` is the honest default: leaving it off means the extremes are the generator's, not an imposed amplification |
| `extreme_prob_threshold` | `0.95` | default accepted deliberately | Inert while `exaggerate_extremes` is `FALSE` |
| `extreme_k` | `1.2` | default accepted deliberately | Inert while `exaggerate_extremes` is `FALSE` |
| `precip_cap_mm_day` | `NULL` | default accepted deliberately | A physical bound on perturbed precip. Uncapped is correct for a stress test: capping would silently truncate the very tail the experiment exists to probe |
| `precip_floor_mm_day` | `NULL` | default accepted deliberately | Same reasoning at the dry end |
| `precip_cap_quantile` | `NULL` | default accepted deliberately | Same, expressed as a quantile |
| `scale_var_with_mean` | `TRUE` | default accepted deliberately | How the perturbation is conditioned. We pass `precip_var_factor` explicitly from `st_<m>.csv`, so the variance axis is already under our control; changing the conditioning would reinterpret that column |
| `enforce_target_mean` | `TRUE` | default accepted deliberately | The design table states a mean change per design point, so the perturbation enforcing it is what makes the response surface's axis mean what it says |
| `verbose` | `FALSE` | default accepted deliberately | Same logging reason as the generate side |

---

## What changed, concretely

**Config template** (`config/templates/weathergen_config.yml`):

| before | after |
| --- | --- |
| `evaluate.model: TRUE` | `save.plots: TRUE` |
| `evaluate.grid.num: 20` | *(removed)* |
| — | `pet.method: hargreaves` |

Both removed keys were **dead** — v1.2.0 reaches neither, so no behaviour is
lost. `save.plots` is the same intent (`# Should performance plots be
generated?`) wired to the argument that actually controls it, and it defaults
`TRUE`, so every tracked config keeps today's behaviour. This is a checked-in
example config key change (`naming.md` §7 tier 2), recorded here and in
`migration_indicator-tables.md`.

**Pinned:** `save.plots` and `pet.method` join `_WG3_GWS_KEYS`, so
`validate_wg3` fails if either goes missing from the generated config. Without
that, a key could silently vanish, the R would read NULL, and weathergenr would
take its own default again — exactly the state C34 exists to end, restored with
nothing noticing.

## Still open, deliberately

- **F16's second half.** Whether the generator's `pet` is used at all before
  rule 3.14 recomputes it. Removing the duplicate work needs a run that shows
  the first result discarded; surfacing the method does not settle it.
- **F15's premise.** Whether `apply_climate_perturbations` is stochastic at all.
  Passing a seed is right either way, so this did not block, but it is unverified.
- **The three stress dimensions** (occurrence frequency, extreme intensification,
  spell length). Already installed and working upstream; C28's writer refuses
  them today by design. They are the strongest argument for widening the design
  table, and they belong to that milestone.
