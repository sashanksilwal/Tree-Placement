#!/usr/bin/env python3
"""
prepare_aoi_data.py — Per-AOI data prep driver for the 3-AOI expansion.

For each (city_id, aoi) in aoi_selections.json:
  - aoi_1 of an existing 239-city: symlink to existing rasters (fast, no download)
  - All other AOIs: call download_city() with the AOI's lon/lat, writing to
    <data_dir>/<city_id>/aoi_<n>/

Idempotent: skips AOIs where DEM.tif, Building_DSM.tif, CDSM.tif, landcover.tif
are all present.

Usage:
  python prepare_aoi_data.py --aoi-selections data/aoi_selections.json
  python prepare_aoi_data.py --only-city 3,5,34   # subset of city IDs
  python prepare_aoi_data.py --only-aoi 1          # aoi_1 symlinks only (fast smoke test)
  python prepare_aoi_data.py --workers 4           # parallel downloads
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_nature_cities import download_city

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("UTHERM_DATA_DIR", "data/300_cities_data"))
DEFAULT_AOI_JSON = Path("data/aoi_selections.json")
DEFAULT_CITIES = Path("data/cities.csv")

AOI_FILES = ["DEM.tif", "Building_DSM.tif", "CDSM.tif", "landcover.tif"]
# Files/dirs that are shared across AOIs in the same city (not AOI-specific)
CITY_SHARED_FILES = ["met.txt", "metadata.json"]
# Shared only for aoi_1 of EXISTING cities — gives the analysis code a uniform
# <city>/aoi_<n>/output/ path regardless of whether n==1 or n==2,3.
# For new arid cities aoi_1 gets its own fresh output/ dir (no symlink).
AOI1_OUTPUT_SHARED = True

# Below this byte size, a real raster is empty/corrupt (PROJ-failed write
# produces ~12KB compressed all-zero TIFFs vs ~1-5MB for valid 1200×1200).
RASTER_VALID_MIN_BYTES = 100_000


def city_complete(city_dir: Path) -> bool:
    """All four rasters exist AND the DEM is not a corruption stub.

    A previous failed run (PROJ DB missing) produced ~12KB DEM TIFFs that
    are technically valid GeoTIFFs but contain only nodata. We treat any
    DEM under RASTER_VALID_MIN_BYTES as incomplete so the next run will
    redo it. Following the symlink for aoi_1's DEM (which points at a
    real ~3MB raster) still resolves correctly.
    """
    for fname in AOI_FILES:
        path = city_dir / fname
        if not path.exists():
            return False
    dem_path = city_dir / "DEM.tif"
    try:
        size = dem_path.stat().st_size
    except OSError:
        return False
    return size >= RASTER_VALID_MIN_BYTES


def symlink_shared_files(city_root: Path, aoi_dir: Path):
    """Symlink met.txt and metadata.json from city root into the AOI dir.

    These are shared across AOIs within a city — same ERA5 met data, same
    metadata block (except the AOI-specific lon/lat is stored per-AOI in the
    raster metadata itself).
    """
    for fname in CITY_SHARED_FILES:
        src = city_root / fname
        dst = aoi_dir / fname
        if src.exists() and not (dst.exists() or dst.is_symlink()):
            os.symlink(f"../{fname}", dst)


def symlink_aoi1(city_id: int, data_dir: Path) -> bool:
    """Create <city_id>/aoi_1/ with symlinks to <city_id>/{DEM,Building_DSM,CDSM,landcover}.tif.

    Returns True if symlinks now exist (freshly made or already present).
    Returns False if the source city dir doesn't have the expected rasters.
    """
    city_root = data_dir / str(city_id)
    aoi_dir = city_root / "aoi_1"

    def _link_output():
        if AOI1_OUTPUT_SHARED and (city_root / "output").exists():
            out_link = aoi_dir / "output"
            if not (out_link.exists() or out_link.is_symlink()):
                os.symlink("../output", out_link)

    if city_complete(aoi_dir):
        # Ensure shared files and output link are idempotently present
        symlink_shared_files(city_root, aoi_dir)
        _link_output()
        return True

    missing = [f for f in AOI_FILES if not (city_root / f).exists()]
    if missing:
        log.warning(f"City {city_id}: cannot create aoi_1 symlinks, missing {missing}")
        return False

    aoi_dir.mkdir(parents=True, exist_ok=True)
    for fname in AOI_FILES:
        dst = aoi_dir / fname
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(f"../{fname}", dst)

    symlink_shared_files(city_root, aoi_dir)
    _link_output()

    log.info(f"City {city_id}: symlinked aoi_1 → existing rasters")
    return True


def download_aoi(city_base: dict, aoi: dict, data_dir: Path) -> tuple:
    """Download rasters for one AOI. Returns (city_id, aoi_id, ok, err)."""
    cid = int(city_base["id"])
    aoi_id = int(aoi["aoi_id"])

    city_for_aoi = dict(city_base)
    city_for_aoi["lon"] = float(aoi["lon"])
    city_for_aoi["lat"] = float(aoi["lat"])

    aoi_dir = data_dir / str(cid) / f"aoi_{aoi_id}"
    if city_complete(aoi_dir):
        return cid, aoi_id, True, "already complete"

    try:
        download_city(city_for_aoi, data_dir, aoi_id=aoi_id)
        # Share met.txt across AOIs within same city
        symlink_shared_files(data_dir / str(cid), aoi_dir)
        return cid, aoi_id, True, None
    except Exception as e:
        return cid, aoi_id, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Per-AOI data prep driver")
    parser.add_argument("--aoi-selections", type=Path, default=DEFAULT_AOI_JSON)
    parser.add_argument("--cities", type=Path, default=DEFAULT_CITIES,
                        help="CSV with id, name, lon, lat (base metadata per city)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Root dir where <city_id>/ folders live")
    parser.add_argument("--only-city", type=str, default=None,
                        help="Comma-separated city IDs to restrict to")
    parser.add_argument("--only-aoi", type=str, default=None,
                        help="Comma-separated AOI IDs (1,2,3) to restrict to")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel download workers for non-symlink AOIs")
    parser.add_argument("--worker", type=int, default=None,
                        help="0-indexed shard number for distributed runs across servers")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards across all servers (default: 1, no sharding)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be done without executing")
    args = parser.parse_args()

    with open(args.aoi_selections) as f:
        selections = json.load(f)
    cities = pd.read_csv(args.cities).set_index("id").to_dict(orient="index")

    only_city = {int(s) for s in args.only_city.split(",")} if args.only_city else None
    only_aoi = {int(s) for s in args.only_aoi.split(",")} if args.only_aoi else None

    symlink_plan = []   # (city_id,) — aoi_1 for existing 239 cities
    download_plan = []  # (city_base_dict, aoi_dict)

    for cid_str, aois in selections.items():
        cid = int(cid_str)
        if only_city is not None and cid not in only_city:
            continue

        # Base city metadata (name, state, lon, lat of the recorded centroid)
        if cid not in cities:
            raise KeyError(f"city {cid} is in aoi_selections but not in the cities CSV")
        city_base = {
            "id": cid,
            "name": cities[cid]["name"],
            "lon": cities[cid]["lon"],
            "lat": cities[cid]["lat"],
        }

        city_root = args.data_dir / str(cid)
        root_has_rasters = all((city_root / f).exists() for f in AOI_FILES)

        for aoi in aois:
            aid = int(aoi["aoi_id"])
            if only_aoi is not None and aid not in only_aoi:
                continue

            aoi_dir = city_root / f"aoi_{aid}"

            # aoi_1 + existing city root rasters → symlink only when the picker
            # flagged this AOI as the original metro centroid. For rescued cities
            # where the picker moved aoi_1 to a non-centroid location
            # (is_original=False), we must do a fresh download at the new lon/lat.
            if aid == 1 and root_has_rasters and aoi.get("is_original", True):
                symlink_plan.append(cid)
                continue

            if city_complete(aoi_dir):
                continue  # already done

            download_plan.append((city_base, aoi))

    # Shard the download plan across servers (symlinks are cheap, run on each)
    if args.worker is not None and args.num_shards > 1:
        original_n = len(download_plan)
        download_plan = [
            t for i, t in enumerate(download_plan)
            if i % args.num_shards == args.worker
        ]
        log.info(f"Shard {args.worker}/{args.num_shards}: "
                 f"{len(download_plan)}/{original_n} downloads")

    log.info(f"Plan: {len(symlink_plan)} aoi_1 symlinks + {len(download_plan)} downloads")

    if args.dry_run:
        for cid in symlink_plan[:5]:
            log.info(f"  SYMLINK city {cid}")
        for cb, aoi in download_plan[:5]:
            log.info(f"  DOWNLOAD city {cb['id']} aoi_{aoi['aoi_id']} "
                     f"at ({aoi['lon']:.4f}, {aoi['lat']:.4f})")
        log.info("(dry-run — no changes made)")
        return

    # Symlinks first (fast)
    failure_count = 0
    for cid in symlink_plan:
        if not symlink_aoi1(cid, args.data_dir):
            failure_count += 1

    # Downloads
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_aoi, cb, aoi, args.data_dir)
                       for cb, aoi in download_plan]
            ok_count, download_failures = 0, 0
            for fut in as_completed(futures):
                cid, aid, ok, err = fut.result()
                if ok:
                    ok_count += 1
                    log.info(f"[{ok_count + download_failures}/{len(download_plan)}] "
                             f"City {cid} aoi_{aid}: OK ({err})")
                else:
                    download_failures += 1
                    log.error(f"[{ok_count + download_failures}/{len(download_plan)}] "
                              f"City {cid} aoi_{aid}: FAIL — {err}")
            failure_count += download_failures
            log.info(f"Downloads: {ok_count} OK, {download_failures} FAILED")
    else:
        for i, (cb, aoi) in enumerate(download_plan):
            cid, aid, ok, err = download_aoi(cb, aoi, args.data_dir)
            status = "OK" if ok else f"FAIL: {err}"
            log.info(f"[{i+1}/{len(download_plan)}] City {cid} aoi_{aid}: {status}")
            if not ok:
                failure_count += 1

    if failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
