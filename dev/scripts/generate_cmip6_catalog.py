"""Regenerate ``config/catalogs/cmip6_data.yml`` from a live crawl of gs://cmip6.

Why a live crawl and not the published index: ``pangeo-cmip6.csv`` (last updated
2022-06) is incomplete at store level — e.g.
``ScenarioMIP/NCC/NorESM2-LM/ssp585/r1i1p1f1/Amon/pr/gn/`` exists in the bucket
but has no row in it. Directory listing is ground truth.

Why one entry per (model, scenario) instead of the previous scenario-level
entries with a shared ``member`` placeholder: hydromt expands ``placeholders``
into the full cross-product, so a shared member list would register
``(model, member)`` combinations that do not exist. ``get_stats_climate_proj.py``
guards with ``if entry in data_catalog.sources`` and silently emits an empty
dataset otherwise — a claim of existence defeats that guard. One entry per model
with that model's exact member list keeps the guard meaningful.

Usage (from the repo root, inside pixi)::

    python dev/scripts/generate_cmip6_catalog.py --out config/catalogs/cmip6_data.yml
    python dev/scripts/generate_cmip6_catalog.py --dry-run      # report only

Not part of a run: this is repository maintenance (see AGENTS.md, "Three homes
for executables").
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")

import gcsfs  # noqa: E402  (must follow the env var above)

# Experiments the toolbox exposes: CMIP historical + the eight ScenarioMIP SSPs.
EXPERIMENTS = {
    "CMIP": ["historical"],
    "ScenarioMIP": [
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
    ],
}
# Monthly atmosphere. WF2 reads this table only; changing it changes every URI.
TABLE = "Amon"
# A member is exposed only if it carries both, i.e. exactly what WF2 needs
# (`variables: [precip, temp]`). Members with tas but no pr are dropped.
REQUIRED_VARS = frozenset({"pr", "tas"})

# Rendered verbatim into the first entry as a YAML anchor; every later entry
# pulls it in with a merge key. Mirrors the pre-2026-07 catalog byte for byte
# except for the uri/placeholders, which are per-entry.
DEFAULTS_BLOCK = """  data_type: RasterDataset
  uri: {uri}
  driver:
    name: raster_xarray
    options:
      drop_variables:
      - time_bnds
      - lat_bnds
      - lon_bnds
      - bnds
      decode_times: true
      preprocess: harmonise_dims
      consolidated: true
      ext_override: .zarr
    filesystem: gcs
  data_adapter:
    unit_add:
      temp: -273.15
    unit_mult:
      precip: 86400
      press_msl: 0.01
    rename:
      pr: precip
      tas: temp
      rsds: kin
      psl: press_msl
  metadata:
    category: climate
    paper_doi: 10.1175/BAMS-D-11-00094.1
    paper_ref: Taylor et al. 2012
    source_license: CC BY 4.0
    source_url: https://console.cloud.google.com/marketplace/details/noaa-public/cmip6
    source_version: 1.3.1
    crs: 4326
"""


def crawl(fs: gcsfs.GCSFileSystem) -> dict[tuple[str, str, str], list[str]]:
    """Return {(activity, "institution/source", experiment): [members]}."""

    def ls(path: str) -> list[str]:
        try:
            return [p.split("/")[-1] for p in fs.ls(path)]
        except FileNotFoundError:
            return []

    from concurrent.futures import ThreadPoolExecutor

    targets = []
    for activity, experiments in EXPERIMENTS.items():
        for institution in ls(f"cmip6/CMIP6/{activity}"):
            for source in ls(f"cmip6/CMIP6/{activity}/{institution}"):
                for experiment in experiments:
                    targets.append((activity, f"{institution}/{source}", experiment))
    print(f"candidate (model, experiment) pairs: {len(targets)}", flush=True)

    def members(target):
        activity, model, experiment = target
        return target, ls(f"cmip6/CMIP6/{activity}/{model}/{experiment}")

    with ThreadPoolExecutor(16) as pool:
        listed = list(pool.map(members, targets))

    def usable(item):
        (activity, model, experiment), member = item
        present = set(ls(f"cmip6/CMIP6/{activity}/{model}/{experiment}/{member}/{TABLE}"))
        return (activity, model, experiment), member, REQUIRED_VARS <= present

    pairs = [(t, m) for t, ms in listed for m in ms]
    print(f"(model, experiment, member) triples to check: {len(pairs)}", flush=True)
    with ThreadPoolExecutor(24) as pool:
        checked = list(pool.map(usable, pairs))

    out: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for key, member, ok in checked:
        if ok:
            out[key].append(member)
    return {k: sorted(v, key=member_sort_key) for k, v in sorted(out.items()) if v}


def member_sort_key(member: str) -> tuple:
    """Sort r2i1p1f1 after r1i1p1f1 rather than lexicographically (r10 < r2)."""
    import re

    m = re.match(r"r(\d+)i(\d+)p(\d+)f(\d+)$", member)
    return (0, *map(int, m.groups())) if m else (1, 0, 0, 0, 0, member)


def render(inventory: dict[tuple[str, str, str], list[str]], crawled_on: str) -> str:
    total = sum(len(v) for v in inventory.values())
    lines = [
        "# GENERATED FILE — do not hand-edit.",
        "# Regenerate with: python dev/scripts/generate_cmip6_catalog.py",
        "#",
        "# One entry per (model, scenario). The `member` placeholder lists exactly the",
        "# members that exist in the bucket with both `pr` and `tas` at Amon, so a",
        "# source name resolving means the store is really there — which is what",
        "# get_stats_climate_proj.py's `if entry in data_catalog.sources` guard relies",
        "# on. Consumers of `kin` (rsds) or `press_msl` (psl) may find a listed member",
        "# lacks those variables; only pr/tas presence is guaranteed.",
        "meta:",
        f"  version: {crawled_on[:7].replace('-', '.')}",
        "  generated_by: dev/scripts/generate_cmip6_catalog.py",
        f"  crawled_on: {crawled_on}",
        "  source: gs://cmip6 (live directory listing, not pangeo-cmip6.csv)",
        f"  entries: {len(inventory)}",
        f"  sources: {total}",
    ]

    first = True
    for (activity, model, experiment), members in inventory.items():
        key = f"cmip6_{model}_{experiment}_{{member}}"
        uri = (
            f"gs://cmip6/CMIP6/{activity}/{model}/{experiment}"
            "/{member}/" + TABLE + "/{variable}/*/*"
        )
        if first:
            lines.append(f"{key}: &cmip6_amon")
            lines.append(DEFAULTS_BLOCK.format(uri=uri).rstrip("\n"))
            first = False
        else:
            lines.append(f"{key}:")
            lines.append("  <<: *cmip6_amon")
            lines.append(f"  uri: {uri}")
        lines.append("  placeholders:")
        lines.append("    member:")
        lines.extend(f"    - {m}" for m in members)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("config/catalogs/cmip6_data.yml"),
        help="catalog to write (default: config/catalogs/cmip6_data.yml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    fs = gcsfs.GCSFileSystem(token="anon")
    inventory = crawl(fs)

    by_experiment: dict[str, int] = defaultdict(int)
    for (_, _, experiment), members in inventory.items():
        by_experiment[experiment] += len(members)
    print("\nmembers per experiment:")
    for experiment, n in by_experiment.items():
        models = sum(1 for k in inventory if k[2] == experiment)
        print(f"  {experiment:12s} {models:3d} models  {n:4d} members")

    union = sorted({m for v in inventory.values() for m in v}, key=member_sort_key)
    first_realizations = [m for m in union if m.startswith("r1i1")]
    print(f"\ndistinct member labels: {len(union)}")
    print(f"first-realization labels (config `members:` union): {first_realizations}")

    text = render(inventory, crawled_on=date.today().isoformat())
    print(f"\nrendered {len(inventory)} entries / {text.count(chr(10))} lines")
    if args.dry_run:
        return
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
