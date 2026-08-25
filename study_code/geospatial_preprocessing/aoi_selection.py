#!/usr/bin/env python3
"""
aoi_selection.py — Pick 3 morphology-contrasted AOIs per city for the
within-city variance analysis.

Reuses find_lcz6_patches() from prepare_nature_cities.py to enumerate all
LCZ-6 patches in CONUS, then for each city in cities_valid_239.csv:

  1. Find all LCZ-6 patches within SEARCH_RADIUS_KM of the city center.
  2. aoi_1 = the patch currently in use (matches city center within tolerance)
     — this preserves the existing simulation for that AOI.
  3. aoi_2, aoi_3 = remaining patches picked to maximize pairwise separation
     and size diversity, respecting MIN_SEPARATION_KM between any two AOIs.
  4. Emit aoi_selections.json with one entry per city.

Output schema:
  {
    "<city_id>": [
      {"aoi_id": 1, "lon": ..., "lat": ..., "patch_pixels": ..., "is_original": true},
      {"aoi_id": 2, ...},
      {"aoi_id": 3, ...}
    ],
    ...
  }

Usage:
  python aoi_selection.py --cities cities_valid_239.csv
  python aoi_selection.py --cities cities_valid_239.csv --prototype 3,5,13,44  # subset
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_nature_cities import LCZ_OPEN_LOWRISE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NATURE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_LCZ = NATURE_DIR / "lcz_filter_v3.tif"
DEFAULT_CITIES = NATURE_DIR / "cities_valid_239.csv"
DEFAULT_OUTPUT = NATURE_DIR / "aoi_selections.json"

SEARCH_RADIUS_KM = 25.0        # half-width of the per-city bounding box
MIN_SEPARATION_KM = 1.5        # minimum centroid-to-centroid spacing between AOIs
PATCH_MIN_PIXELS = 100         # same threshold as prepare_nature_cities (~1 km²)
CANDIDATE_MIN_PIXELS = 500     # floor for candidate patches (≥5 ha)
CURRENT_AOI_TOLERANCE_KM = 0.5 # how close a patch must be to the recorded city
                               # centroid to count as "the original AOI"
KMEANS_SUBSAMPLE = 20000       # max LCZ-6 pixels to feed k-means per city
MIN_CLUSTER_PIXELS = 500       # k-means cluster must have at least this many
                               # LCZ-6 pixels to be a valid AOI (5 ha)


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def extract_lcz6_pixels(lcz_path, city_lon, city_lat, half_radius_km):
    """Return (lons, lats, original_patch_centroid) for LCZ-6 pixels within a
    2 * half_radius_km bounding box of (city_lon, city_lat).

    Works uniformly whether the city's LCZ-6 is monolithic (Detroit) or
    fragmented (Denver) — we get every LCZ-6 pixel's location.
    """
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(city_lat))
    half_lat = half_radius_km / km_per_deg_lat
    half_lon = half_radius_km / km_per_deg_lon

    with rasterio.open(lcz_path) as src:
        west = city_lon - half_lon
        east = city_lon + half_lon
        south = city_lat - half_lat
        north = city_lat + half_lat
        win = from_bounds(west, south, east, north, src.transform)
        data = src.read(1, window=win)
        win_transform = src.window_transform(win)

    mask = (data == LCZ_OPEN_LOWRISE)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.array([]), np.array([])

    lons, lats = rasterio.transform.xy(win_transform, rows, cols)
    return np.asarray(lons), np.asarray(lats)


def kmeans_cluster_centers(lons, lats, n_clusters, seed=42):
    """Simple k-means on (lon, lat) in km-projected space.

    We project to an equirectangular km frame so Euclidean distance is in km.
    """
    if len(lons) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int), np.array([])

    from sklearn.cluster import KMeans
    lat0 = float(np.mean(lats))
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat0))
    xs = (lons - np.mean(lons)) * km_per_deg_lon
    ys = (lats - np.mean(lats)) * km_per_deg_lat
    X = np.column_stack([xs, ys])

    n_clusters = min(n_clusters, len(lons))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    labels = km.labels_
    counts = np.bincount(labels, minlength=n_clusters)

    # Cluster centers back to lon/lat (center of mass of member pixels)
    centers = np.zeros((n_clusters, 2))
    for k in range(n_clusters):
        mask_k = labels == k
        centers[k] = [float(np.mean(lons[mask_k])), float(np.mean(lats[mask_k]))]
    return centers, labels, counts


def _kmeans_fill(picked, lons, lats, n_aoi):
    """Fill picked[] up to n_aoi using k-means on LCZ-6 pixels that lie
    ≥ MIN_SEPARATION_KM from every already-picked AOI. Mutates and returns picked."""
    mask = np.ones(len(lons), dtype=bool)
    for p in picked:
        d = haversine_km(lons, lats, p["lon"], p["lat"])
        mask &= d >= MIN_SEPARATION_KM
    remaining_lons = lons[mask]
    remaining_lats = lats[mask]
    if len(remaining_lons) < MIN_CLUSTER_PIXELS:
        return picked

    subsample_ratio = 1.0
    if len(remaining_lons) > KMEANS_SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(remaining_lons), size=KMEANS_SUBSAMPLE, replace=False)
        subsample_ratio = len(remaining_lons) / KMEANS_SUBSAMPLE
        remaining_lons = remaining_lons[idx]
        remaining_lats = remaining_lats[idx]

    # Over-cluster by 2 so we have spare candidates if some fail MIN_SEPARATION.
    n_remaining = n_aoi - len(picked)
    k = min(n_remaining + 2, max(n_remaining, len(remaining_lons)))
    centers, labels, counts = kmeans_cluster_centers(remaining_lons, remaining_lats, k)
    if len(centers) == 0:
        return picked
    true_counts = (counts * subsample_ratio).astype(int)

    for i in np.argsort(-true_counts):
        if len(picked) >= n_aoi:
            break
        if true_counts[i] < MIN_CLUSTER_PIXELS:
            continue
        if picked:
            d = haversine_km(centers[i, 0], centers[i, 1],
                             np.array([p["lon"] for p in picked]),
                             np.array([p["lat"] for p in picked]))
            if float(np.min(d)) < MIN_SEPARATION_KM:
                continue
        picked.append({
            "aoi_id": len(picked) + 1,
            "lon": float(centers[i, 0]),
            "lat": float(centers[i, 1]),
            "patch_pixels": int(true_counts[i]),
            "is_original": False,
        })
    return picked


def pick_aois_kmeans(lcz_path, city_lon, city_lat, n_aoi=3, allow_free_aoi1=True):
    """Pick up to n_aoi AOIs.

    Primary pass (preserves Phase 2 reuse): aoi_1 is fixed at the recorded metro
    centroid, aoi_2..n come from k-means on LCZ-6 pixels outside the aoi_1 buffer.

    Fallback (allow_free_aoi1=True): if the primary pass returns < n_aoi — which
    happens when the metro centroid sits inside the main LCZ-6 blob and its 1.5 km
    exclusion circle blocks too much of the eligible area — retry from scratch
    with pure k-means (no fixed aoi_1). The fallback aoi_1 is flagged
    is_original=False so downstream steps know it needs fresh data prep.
    """
    lons, lats = extract_lcz6_pixels(lcz_path, city_lon, city_lat, SEARCH_RADIUS_KM)
    if len(lons) == 0:
        return []

    # Primary pass: aoi_1 pinned to metro centroid
    picked = [{
        "aoi_id": 1,
        "lon": float(city_lon),
        "lat": float(city_lat),
        "patch_pixels": int(len(lons)),
        "is_original": True,
    }]
    picked = _kmeans_fill(picked, lons, lats, n_aoi)
    if len(picked) >= n_aoi or not allow_free_aoi1:
        return picked

    # Fallback pass: free k-means (no fixed aoi_1)
    free_picked = []
    free_picked = _kmeans_fill(free_picked, lons, lats, n_aoi)
    # Only replace if fallback strictly improves on primary
    if len(free_picked) > len(picked):
        return free_picked
    return picked


def pick_aois_for_city(city_lon, city_lat, patches_nearby, n_aoi=3):
    """Pick up to n_aoi LCZ-6 patches for one city.

    Strategy: "top-N largest LCZ-6 neighborhoods, spatially separated."
      aoi_1 = patch closest to the recorded city centroid (preserves original)
      aoi_2..n = iterate through remaining candidates sorted by patch size
                 descending; accept if separation >= MIN_SEPARATION_KM from
                 all already-picked AOIs.

    This picks the 3 biggest real LCZ-6 neighborhoods in the city rather
    than tiny fringe blips at maximum separation.

    Returns list of dicts with aoi_id, lon, lat, patch_pixels, is_original.
    """
    if len(patches_nearby) == 0:
        return []

    # Drop patches below the candidate floor (avoid 1-ha blips as AOI centers)
    patches_nearby = [p for p in patches_nearby if p["pixels"] >= CANDIDATE_MIN_PIXELS]
    if len(patches_nearby) == 0:
        return []

    lons = np.array([p["lon"] for p in patches_nearby])
    lats = np.array([p["lat"] for p in patches_nearby])
    pixels = np.array([p["pixels"] for p in patches_nearby])

    dist_to_center = haversine_km(city_lon, city_lat, lons, lats)

    # aoi_1: patch closest to the recorded city centroid. For the existing 239
    # cities, this will be the same patch that the original pipeline selected.
    nearest_idx = int(np.argmin(dist_to_center))
    is_original = dist_to_center[nearest_idx] <= CURRENT_AOI_TOLERANCE_KM
    picked = [{
        "aoi_id": 1,
        "lon": float(lons[nearest_idx]),
        "lat": float(lats[nearest_idx]),
        "patch_pixels": int(pixels[nearest_idx]),
        "is_original": bool(is_original),
    }]

    # Remaining candidates sorted by patch size descending
    order = np.argsort(-pixels)
    for i in order:
        if i == nearest_idx:
            continue
        if len(picked) >= n_aoi:
            break

        picked_lons = np.array([p["lon"] for p in picked])
        picked_lats = np.array([p["lat"] for p in picked])
        d = haversine_km(lons[i], lats[i], picked_lons, picked_lats)
        if float(np.min(d)) < MIN_SEPARATION_KM:
            continue

        picked.append({
            "aoi_id": len(picked) + 1,
            "lon": float(lons[i]),
            "lat": float(lats[i]),
            "patch_pixels": int(pixels[i]),
            "is_original": False,
        })

    return picked


def build_selections(cities_df, lcz_path, n_aoi=3):
    """For each city, pick AOIs via k-means on LCZ-6 pixels. Returns dict keyed by city_id."""
    selections = {}
    fallbacks = 0
    total_aois = 0

    for _, row in cities_df.iterrows():
        city_id = int(row["id"])
        city_lon = float(row["lon"])
        city_lat = float(row["lat"])

        picks = pick_aois_kmeans(lcz_path, city_lon, city_lat, n_aoi=n_aoi)

        selections[str(city_id)] = picks
        total_aois += len(picks)
        if len(picks) < n_aoi:
            fallbacks += 1
            log.warning(f"City {city_id} ({row['name']}): only {len(picks)} AOIs")

    log.info(f"Built selections for {len(selections)} cities: "
             f"{total_aois} AOIs total, {fallbacks} cities below target n_aoi={n_aoi}")
    return selections


def main():
    parser = argparse.ArgumentParser(description="Pick 3 AOIs per city for within-city variance analysis")
    parser.add_argument("--cities", type=Path, default=DEFAULT_CITIES,
                        help="CSV with id, lon, lat, name")
    parser.add_argument("--lcz", type=Path, default=DEFAULT_LCZ,
                        help="Path to lcz_filter_v3.tif")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON file")
    parser.add_argument("--n-aoi", type=int, default=3,
                        help="Target AOIs per city (default 3)")
    parser.add_argument("--prototype", type=str, default=None,
                        help="Comma-separated city IDs to restrict (for Phase 4 QA)")
    parser.add_argument("--cached-patches", type=Path, default=None,
                        help="Optional path to cached patches JSON (skips scipy.ndimage run)")
    args = parser.parse_args()

    cities_df = pd.read_csv(args.cities)
    if args.prototype:
        ids = {int(s) for s in args.prototype.split(",")}
        cities_df = cities_df[cities_df["id"].isin(ids)]
        log.info(f"Prototype subset: {list(cities_df['id'])}")

    # k-means path: we don't need pre-computed patch centroids, we read
    # LCZ-6 pixels per city on demand.
    log.info(f"Using k-means on LCZ-6 pixels per city (LCZ raster: {args.lcz})")
    selections = build_selections(cities_df, args.lcz, n_aoi=args.n_aoi)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(selections, f, indent=2)
    log.info(f"Wrote AOI selections to {args.output}")


if __name__ == "__main__":
    main()
