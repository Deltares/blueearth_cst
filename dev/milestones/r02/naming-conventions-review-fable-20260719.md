# R02 revised-design review — Fable (2026-07-19)

**Subject.** `dev/milestones/r02/naming-conventions-design.md` (working tree, uncommitted; `git diff --stat` shows 50 insertions / 23 deletions against the committed version), reviewed against `dev/milestones/r02/naming-conventions-review.md`, `dev/milestones/r02/naming-conventions-review-gpt-20260718.md`, `dev/roadmap.md`, `AGENTS.md`, and targeted repo sources.

---

## 1. Verification of the six required corrections

**C1 — §1/§8 filename contradiction: applied correctly.**
`dev/milestones/r02/naming-conventions-design.md` §1 (lines 71–73) now reads "snake_case for variables, functions, and modules (MUST). File names are governed by class — see §8, not this universal rule." The decisions table (line 51) also omits "files." §8's class table is unchanged and consistent. No residual "snake_case for files" claim anywhere in the doc.

**C2 — `st_num` range and fold ownership: applied correctly and verified against the Snakefile.**
§4 (line 130) defines `1..stress_test_count` = perturbed, `0` = reserved unperturbed baseline (`cst_0`), Wflow-run only when `run_historical` sets `ST_START = 0`. This matches `Snakefile_climate_experiment` exactly: perturbation CSVs span `1..ST_NUM` (line 84), `cst_0` is the unperturbed realization (line 122), and `st_num2` spans `ST_START..ST_NUM` with `ST_START = 0 if run_hist else 1` (lines 38–39, 180). The precision of "run through Wflow only when" is correct — `cst_0` is always *generated*, conditionally *run*. The fold is now assigned to **R5** alone (lines 136–139), consistent with `dev/roadmap.md` R5 (lines 366–384, which explicitly owns `Snakefile_climate_experiment` and the ST_NUM grid helper). The design's paraphrase "no milestone touches another's Snakefile" is a fair rendering of roadmap line 468.

**C3 — "true constants" rewording: applied, but the fix introduced a small factual error.**
Line 53 adopts the GPT-proposed formulation ("fixed, non-config-derived values / lookup tables not reassigned or mutated at runtime") — good. But the appended parenthetical says "Some are not language-immutable (`WFLOW_VARS` is a dict, `XDIMS`/`YDIMS` are tuples)." Tuples **are** language-immutable in Python; grouping `XDIMS`/`YDIMS` (`src/get_change_climate_proj.py:212–213`) under "not language-immutable" alongside the genuinely mutable `WFLOW_VARS` dict (`src/setup_gauges_and_outputs.py:13`) is wrong as written. `VOLATILE_NC_ATTRS` is a `frozenset` (`dev/scripts/check_baseline.py:43`). One-clause fix: only `WFLOW_VARS` belongs in the non-immutable parenthetical. Low severity — the rule itself (intent, not enforcement) is right.

**C4 — three-tier domain-identifier taxonomy: applied; tier contents verified; one new §6↔§7 tension (see finding N1).**
§6 (lines 184–218) implements the three tiers with per-tier rationale. Tier contents check out against the repo: the semantic-label vs CSDMS-ID distinction matches `src/setup_gauges_and_outputs.py:10–20` (`actual evapotranspiration` → `land_surface__evapotranspiration_volume_flux`); `Qstats`/`Tlow`/`Tpeak` are BlueEarth output/config surfaces (`Snakefile_climate_experiment:51,182,188–189`); catalog source pattern `cmip6_<model>_<scenario>_<member>` is real (`config/cmip6_data.yml:3,65,106,154`), as are `INM/INM-CM5-0` IDs (line 45).

**C5 — YAML rule keyed on consuming contract: applied correctly.**
Decisions table rows (lines 58–59) and §2 (lines 100–108) both discriminate by consuming contract, with the explicit "even when BlueEarth-generated" carve-out for HydroMT catalogs and the closing sentence "never authorship or whether the file is checked in vs. generated." The old "generated catalogs are BlueEarth-owned snake_case" conflict is gone.

**C6 — commit prefixes: applied correctly.**
Lines 324–331: three `r02:` subjects matching GPT's suggested replacements verbatim, the obsolete "open milestone" commit dropped with an explanatory note, and the note's rationale (design already in `r01-contracts` ancestry) is factually right — the file exists in committed history with only the revision uncommitted. Prefix scheme matches `dev/roadmap.md:488`.

**Summary: 6/6 applied; C3 carries a minor new inaccuracy; C4 sharpens a pre-existing §6/§7 tension (below).**

---

## 2. Independent findings (missed by both prior reviews)

**N1 — §6 tier 1 "preserve verbatim" contradicts §7's rename-with-migration-note list.** §6 tier 1 (lines 189–199) says opaque upstream identifiers — including HydroMT catalog source names — are preserved verbatim because "renaming breaks downstream tools or catalog lookups." Yet §7 (lines 224–229) lists "Data catalog source names in `config/*.yml` (catalog contract)" and "Wflow / HydroMT / CMIP / weathergenr external identifiers" as renameable *with* a migration note. Under the new taxonomy those are tier 1, whose whole point is that no local rename path exists (you cannot migration-note your way out of a CMIP model ID). Also, catalog *source names* are arguably tier 2, not tier 1: for BlueEarth-generated catalogs the *schema* (adapter fields) is HydroMT's, but the source-name strings are BlueEarth-minted lookup keys — the consuming-contract logic of C5, applied consistently, splits name from schema. When authoring `dev/reference/naming.md`, reconcile: tier 1 = not renameable at all; §7's migration-note list should cover tier-2 surfaces only. This is the one substantive new issue; it predates the corrections but the three-tier taxonomy now makes it visible.

**N2 — `my_cfg` is not merely a weak example; it is the live R01 convention.** GPT's nit treated `my_cfg` (line 161) as a made-up placeholder. It is in fact the established identifier in all three Snakefiles (`Snakefile_model_creation:20`, `Snakefile_climate_projections:20`, `Snakefile_climate_experiment:21`). So the guide must make a real decision the design currently dodges: either bless `my_cfg` as the per-Snakefile own-section idiom (defensible — it is uniform across all three) or deprecate-and-grandfather it like `_fn`. Listing it silently as a `_cfg` example does neither.

**N3 — Trivial:** the doc header date is still 2026-05-09 (line 3) despite two review rounds and a substantive revision — worth bumping at commit 1 for provenance. And §4's `rlz_num` row still glosses its range as `1..RLZ_NUM` (grandfathered name) while the `st_num` row uses the future name `stress_test_count`; harmless but inconsistent vocabulary within one table.

Otherwise the revised design is internally consistent, faithful to both prior reviews' accepted recommendations, and its repo claims survived every spot-check I ran (wildcard semantics, constants, catalog names, `Qstats`/`Tlow`/`Tpeak`, dotted weathergenr keys via prior GPT verification, `dev/conventions-review.md` genuinely absent from `dev/`). The nine-section outline remains a sufficient content spec to execute without a separate plan doc.

---

## 3. Unaddressed prior-review items

| Item (GPT review §4–§6) | Status in revised design | Blocking? |
|---|---|---|
| Pointer in `AGENTS.md`, not just `CLAUDE.md` | **Unaddressed.** Verification (line 288) still says "`CLAUDE.md` has a one-line pointer"; `dev/roadmap.md:261` (R2 exit criteria) likewise. Commit 2's subject ("agent-instruction pointers", plural, line 327) hints at it but nothing normative requires `AGENTS.md`. | **Yes-ish — the most important gap.** `AGENTS.md` itself mandates "Author repo instructions **here**, never only in `CLAUDE.md`," and the project `CLAUDE.md` says "Edit `AGENTS.md`, not this file." Executing the design as literally written violates the repo's own instruction-file contract. |
| Docs-only changed-path allowlist | **Unaddressed.** Verification still has only the weaker "No code files modified in R2" (line 290). | No, but cheap and it operationalizes the milestone's core promise; fold into Verification. |
| Pin "suite unchanged" to 51/3/2 | **Unaddressed** (line 289: "unchanged"). The sealed state is 51 passed / 3 skipped / 2 xfailed (`dev/roadmap.md:142,199`). | No. |
| Update `dev/reference/git-conventions.md` at seal | **Partially covered** by commit 3's "durable refs" (line 328); Verification is silent. `dev/reference/git-conventions.md:39–40` still lists `r02-naming` as planned, per its own maintenance rule (lines 56–58). | No. |
| Risk 2: "R2 MUST NOT rename" banner over §9 examples | **Unaddressed** — no banner (line 257ff), though grandfathering is stated three times elsewhere. | No. |
| Nits: `weagen_config_yml` (line 169; label doesn't exist — Snakefile uses `weagen_config`), `precip_plt` in the `_png` row (line 170; real label, `Snakefile_climate_projections:51,154`, but not a `_png` example), `my_cfg` (line 161; see N2), pathlib overstatement (lines 117–118) | **All four unaddressed**, carried verbatim into the revised design. | No individually — but since §5/§3 are the content spec the guide will be transcribed from, fix them before authoring or the errors propagate into `dev/reference/naming.md`. |

None of these blocks *starting* execution except the `AGENTS.md` pointer, which as written directs the deliverable's discoverability edit at the wrong file.

---

## 4. Verdict

**Ready-with-changes (one small edit batch — materially closer to ready than at the GPT round).** All six required corrections are correctly applied and repo-verified; the design is internally consistent, faithful to both reviews, and specific enough to serve as its own plan. What remains is a single pre-execution touch-up to the design (naturally the already-planned commit 1, `r02: tighten naming design after independent review`): (a) change the Verification pointer target to `AGENTS.md` (+ mirror in the roadmap R2 exit criteria) — the only item that would produce a wrong artifact if executed as written; (b) reconcile §6 tier 1 "preserve verbatim" with §7's migration-note list and decide the tier of catalog source names (N1); (c) fix the tuple-immutability clause in the C3 wording (line 53); (d) decide `my_cfg`'s status explicitly (N2); (e) sweep in the four carried-over nits, the allowlist, and the 51/3/2 pin. All are doc-line edits in one file plus two roadmap lines; none reopens a design decision.
