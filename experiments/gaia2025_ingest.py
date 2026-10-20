#!/usr/bin/env python3
"""Download the GAIA 2019–25 per-flight files (vPM-enabled CoCiP labels)
from contrails.org's bucket, keeping only what the model needs.

The bucket holds one parquet per day (~91 GB, 150 columns, all aircraft).
Our corpus is jet aircraft only, as in the original 2019/2021 data, so each
file is streamed, filtered to jets, cut to the columns below, and written to
Dropbox at `GAIA2019-25/{folder}/{day}.parquet`. Reruns skip finished files.

Column notes (checked against the old corpus on 2021-01-15):
  - `aircraft_type` equals the old `aircraft_type_icao` on every flight;
    `atyp_icao_ch_a` does not (blank for ~25% of jets), so it is not used.
  - `ef_sum` is NULL where no persistent contrail forms (old files: 0).
  - flight_ids match the old version exactly.
  - Some year folders also hold a small file for the previous 31 December;
    deduplicate by flight_id when building the corpus.

Requires `gcloud auth login` with an account that can read the bucket.
Usage:
  python experiments/gaia2025_ingest.py [--years 2019 2020 ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DROPBOX_PARQUETS  # noqa: E402

BUCKET = "gs://contrails-301217-gaia-sims-2019-25-stanford"
OUT = DROPBOX_PARQUETS.parent / "GAIA2019-25"
YEARS = [str(y) for y in range(2019, 2026)]

KEEP = [
    # identity and schedule
    "flight_id", "flight_number", "operator_icao", "operator_type",
    "origin_airport_icao", "destination_airport_icao",
    "scheduled_departure_time_utc", "scheduled_arrival_time_utc",
    "first_waypoint_time", "last_waypoint_time",
    # aircraft and engine
    "aircraft_type", "aircraft_engine_type", "aircraft_market_group",
    "aircraft_role", "engine_name", "engine_uid", "n_seats",
    "passenger_load_factor", "aircraft_age_yrs",
    # trajectory quality
    "total_distance_km", "great_circle_distance_km", "trajectory_complete",
    "est_origin_airport", "est_destination_airport",
    # emissions and contrail labels
    "total_fuel_burn", "mean_nvpm_ei_n", "n_ice_per_m_0_mean",
    "total_contrail_length_sac_km", "total_persistent_contrail_length_km",
    "ef_sum", "ef_overlap_sum",
]


def gcloud(*args: str, binary: bool = False):
    r = subprocess.run(["gcloud", "storage", *args], capture_output=True,
                       check=True)
    return r.stdout if binary else r.stdout.decode()


def fetch(uri: str, dest: Path) -> tuple[str, int, int]:
    """Stream one day file (`storage cat`; `storage cp` hangs on finalize
    here), filter to jets, write the slim parquet atomically."""
    for attempt in range(4):
        try:
            raw = gcloud("cat", uri, binary=True)
            t = pq.read_table(io.BytesIO(raw), columns=KEEP)
            n_all = t.num_rows
            t = t.filter(pc.equal(t["aircraft_engine_type"], "Jet"))
            tmp = dest.with_suffix(".tmp")
            pq.write_table(t, tmp, compression="zstd")
            tmp.rename(dest)
            return uri, n_all, t.num_rows
        except Exception as e:  # noqa: BLE001 — network retry
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{uri}: {err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=YEARS)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for y in a.years:
        (OUT / y).mkdir(parents=True, exist_ok=True)
        for uri in gcloud("ls", f"{BUCKET}/{y}/").split():
            dest = OUT / y / (Path(uri).stem + ".parquet")
            if not dest.exists():
                jobs.append((uri, dest))
    print(f"{len(jobs)} files to fetch -> {OUT}", flush=True)
    done = n_all = n_jet = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, u, d) for u, d in jobs]
        for f in as_completed(futs):
            try:
                _, a_, j_ = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {e}", flush=True)
                continue
            done += 1
            n_all += a_
            n_jet += j_
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"{done}/{len(jobs)} files, {n_jet:,} jet flights "
                      f"of {n_all:,} ({el / 60:.0f} min, "
                      f"~{el / done * (len(jobs) - done) / 60:.0f} min left)",
                      flush=True)


if __name__ == "__main__":
    main()
