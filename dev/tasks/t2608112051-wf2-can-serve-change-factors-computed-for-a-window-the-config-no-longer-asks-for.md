---
title: WF2 can serve change factors computed for a window the config no longer asks for
type: watch-item
area: wf2 projections
origin: t2608111659 rapid rebuild
created: 2026-08-11
updated: 2026-08-11
---

> [!note] Overview
> **What** — raw/ and scalar/ are keyed by (model, scenario, member) and are window-independent; historical_year_range and future_horizons enter only at stage B (2.06 derive_change_factors). If the .snakemake driving a run never saw the build that produced summary/, Snakemake has no metadata for 2.06, the params rerun-trigger cannot fire, and mtime reports summary/ as newer than scalar/. A full run then leaves the OLD change factors in place while reporting success.
> **Why** — Rule 3.01 check_project_consistency compares config SECTIONS, not products, so the tree passes its guard while the CMIP6 overlay describes a different experiment than the config asks for. Measured 2026-08-11 on test_rapid: after a clean run_workflows.py pass the summary still carried 3 models, the far horizon and a 1990-2010 reference window against the config's 2 / mid / 2000-2014. Nothing reported it. Harmless for a stress test -- CMIP6 is a plausibility overlay and never drives WF3 -- but the overlay is read by humans, and a plausibility panel drawn for the wrong horizon is worse than none.
> **Trigger** — Someone reads a change-factor product as current when its provenance.json horizon or reference window disagrees with the config; or WF2 gains a consumer that is not merely an overlay. Fix directions: fold the window and horizon set into series_digest_components so the revalidation catches it, or have 2.06 assert its own provenance.json against the live config the way 3.01 asserts snapshots.
