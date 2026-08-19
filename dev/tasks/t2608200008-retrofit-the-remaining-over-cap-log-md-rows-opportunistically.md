---
title: Retrofit the remaining over-cap LOG.md rows, opportunistically
type: watch-item
area: dev records / ledger
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — 32 `dev/LOG.md` rows still exceed the three-sentence cap
> (`dev/README.md` § The cap on `LOG.md` rows). Shorten one when you are
> already working in its territory, never as a batch.
> **Why** — The cap's real job, stopping future growth, is already done, and
> nobody reads old rows in bulk — they grep for an ID. The retrofit's value is
> legibility; its risk is destroying a lesson that lives nowhere else. That risk
> is not theoretical: retrofitting the first row (`t2608131847`, 563 → 85 words)
> found **2 of its 6 lessons un-homed**, and both were checks that catch a
> notebook cell which ran and produced nothing.
> **Trigger** — see below; this is deliberately not scheduled work.

## The procedure, which is not optional

1. Read the row and list what it asserts.
2. `git grep` each specific **outside `dev/LOG.md`**. A git SHA needs no other
   home — history is its home. A measurement, ruling, threshold or check does.
3. Anything un-homed gets a home FIRST — a `reference/` doc, a test, an ADR,
   `AGENTS.md`, a follow-on note — in the same commit.
4. Only then compress, to at most three short sentences: what closed, why,
   where anything durable now lives.

A row whose detail survives nowhere is not a row to compress; it is a lesson
without a home. That inversion is the whole point.

## Where it stands

Measured 2026-08-20: 49 closure rows, 33 over the cap, median 172 words,
longest was 563. One retrofitted (`t2608131847`, in `12b7d85`, which also moved
its two orphaned checks into `docs/notebooks/README.md`). **32 remain.**

The eight largest are `t2608132341a` (523w), `t2608152230` (518w),
`t2608071216` (472w), `t2608132341` (408w), `t2608071203` (357w),
`t2608130215` (326w), `t2608071206` (312w), `t2608172138` (263w).

A first-pass token sweep suggested most of their specifics already resolve
somewhere outside the ledger, with the apparent exceptions being git SHAs —
which need nothing. Treat that as a hint, not a clearance: it tests whether a
STRING appears elsewhere, not whether a LESSON has a home, and `t2608131847`
passed it while still hiding two orphans.

## Links

[[t2608132100]], the worked example in `dev/README.md` — its dropped
measurements survive in four places, one of them a test.
