#!/usr/bin/env python3
"""Build the pooled 2019 + 2021 feature cache read by every training,
evaluation and figure script (experiments/outputs/_pool_cache_all.parquet,
~52.5M flights, ~2.4 GB, gitignored).

Reads all 47 weekly label shards per year from DROPBOX_PARQUETS, computes
the engineered features with common.add_features(), drops rows with any
non-finite engineered feature or target, and tags each row with its year.
Takes roughly 6 minutes. An existing cache is left alone unless --force.
"""
import argparse
import time

import pandas as pd

from common import OUT, load_year

CACHE = OUT / "_pool_cache_all.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the cache exists")
    args = ap.parse_args()
    if CACHE.exists() and not args.force:
        print(f"{CACHE.name} exists; pass --force to rebuild")
        return

    t0 = time.time()
    parts = []
    for year in (2019, 2021):
        print(f"Loading {year} ...")
        df = load_year(year, None)
        df["year"] = year
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True)
    del parts
    pool["aircraft_type_icao"] = pool["aircraft_type_icao"].astype("category")
    OUT.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(CACHE)
    print(f"Wrote {CACHE.name}: {len(pool):,} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
