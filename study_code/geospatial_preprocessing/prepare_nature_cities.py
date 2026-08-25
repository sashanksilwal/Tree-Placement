#!/usr/bin/env python3
"""
prepare_nature_cities.py — Select US cities with LCZ 6 (Open Low-rise) patches
and download 1.2km x 1.2km DEM, Building DSM, and Canopy DSM from 3DEP COPC.

Steps:
  1. Download global LCZ map (lcz_filter_v3.tif) from Zenodo
  2. Mask to CONUS, find LCZ 6 patches (connected components, min 100 pixels)
  3. Cluster into cities using US Census urban areas
  4. COPC pre-flight: verify 3DEP LiDAR coverage exists at each centroid
  5. Select top N cities by LCZ 6 pixel count
  6. For each city, download 1.2km x 1.2km from COPC:
       - DEM (class 2, ground, mean Z)
       - Building_DSM (classes 1+2+6, max Z)
       - CDSM (classes 3+4+5, max Z minus DEM = canopy height)

Download domain: 1.2km x 1.2km
Analysis domain: inner 800m x 800m (200m shadow/wind buffer on each side)

Output: <output-dir>/<city_id>/
        DEM.tif, Building_DSM.tif, CDSM.tif, metadata.json

Usage:
  python prepare_nature_cities.py --step select   # steps 1-5 -> cities.json
  python prepare_nature_cities.py --step download  # step 6: download all
  python prepare_nature_cities.py --step download --city 42  # single city
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.geometry import box, Point

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import pdal
except ImportError:
    pdal = None

try:
    import pystac_client
    import planetary_computer
except ImportError:
    pystac_client = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# -- Constants ----------------------------------------------------------------

LCZ_URL = "https://zenodo.org/records/8419340/files/lcz_filter_v3.tif"
LCZ_OPEN_LOWRISE = 6

CONUS_WEST, CONUS_SOUTH, CONUS_EAST, CONUS_NORTH = -125.0, 24.0, -66.5, 50.0

RESOLUTION = 1.0  # meters
PATCH_SIZE_M = 1200  # 1.2km download domain
ANALYSIS_SIZE_M = 800  # inner 800m analysis domain

CLASS_GROUND = [2]
CLASS_BUILDING = [1, 2, 6]
CLASS_VEGETATION = [3, 4, 5]
# Extended vegetation classes seen in some USGS 3DEP datasets
# Many datasets don't use standard 3/4/5 — they use 17/18/20 or leave as class 1
CLASS_NOISE = [7, 18]  # Low point / High noise (ASPRS 1.4)

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-lidar-copc"
MS_BUILDINGS_COLLECTION = "ms-buildings"
ESA_WORLDCOVER_COLLECTION = "esa-worldcover"
GAP_FILL_DISTANCE = 50

# ESA WorldCover 2021 v200 class definitions
# https://worldcover2021.esa.int/
ESA_WC_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# Reclassify ESA WorldCover → UTherm land cover codes (1-7)
# UTherm codes: 1=asphalt, 2=roofs, 5=grass, 6=bare soil, 7=water
# Trees (code 10) map to grass because tree canopy is handled by CDSM.tif
ESA_TO_UTHERM = {
    10: 5,   # tree cover → grass (canopy in CDSM)
    20: 5,   # shrubland → grass
    30: 5,   # grassland → grass
    40: 5,   # cropland → grass
    50: 1,   # built-up → asphalt
    60: 6,   # bare/sparse → bare soil
    70: 7,   # snow/ice → water
    80: 7,   # water → water
    90: 5,   # wetland → grass
    95: 5,   # mangroves → grass
    100: 6,  # moss/lichen → bare soil
}

CENSUS_UA_URL = "https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20.zip"
DEFAULT_OUTPUT_DIR = os.environ.get("UTHERM_DATA_DIR", "data/300_cities_data")


# -- COPC Pre-flight ----------------------------------------------------------


def verify_copc_coverage(lon: float, lat: float, min_items: int = 1) -> int:
    """Quick STAC query to check COPC LiDAR coverage at a location.

    Queries a small bounding box (~200m) around the centroid.
    Returns number of COPC items found, 0 if none.
    """
    if pystac_client is None:
        raise ImportError("pystac-client required: pip install pystac-client planetary-computer")

    # Use ~1.2km bbox (matching download domain) to avoid false negatives
    # when centroid falls between COPC flight lines
    delta_lon = 0.01
    delta_lat = 0.01
    bbox = (lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat)

    try:
        catalog = pystac_client.Client.open(
            STAC_API, modifier=planetary_computer.sign_inplace
        )
        search = catalog.search(collections=[COLLECTION], bbox=bbox, limit=5)
        items = list(search.items())
        return len(items)
    except Exception as e:
        log.warning(f"  COPC pre-flight failed for ({lon:.4f}, {lat:.4f}): {e}")
        return 0


# -- Step 1: Download LCZ map ------------------------------------------------


def download_lcz(output_dir: Path) -> Path:
    """Download LCZ filtered GeoTIFF from Zenodo if not already present."""
    lcz_path = output_dir / "lcz_filter_v3.tif"
    if lcz_path.exists():
        log.info(f"LCZ map already exists: {lcz_path}")
        return lcz_path

    log.info("Downloading LCZ map from Zenodo (~1.4 GB)...")
    lcz_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(LCZ_URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(lcz_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)",
                      end="", flush=True)
    print()
    log.info(f"Downloaded LCZ map to {lcz_path}")
    return lcz_path


# -- Step 2: Download Census urban areas -------------------------------------


def download_census_urban_areas(output_dir: Path) -> Path:
    """Download US Census 2020 urban areas shapefile."""
    ua_dir = output_dir / "census_ua"
    ua_shp = ua_dir / "tl_2020_us_uac20.shp"
    if ua_shp.exists():
        log.info(f"Census urban areas already exist: {ua_shp}")
        return ua_shp

    zip_path = output_dir / "tl_2020_us_uac20.zip"
    if not zip_path.exists():
        log.info("Downloading US Census urban areas shapefile...")
        resp = requests.get(CENSUS_UA_URL, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

    import zipfile
    ua_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ua_dir)
    log.info(f"Extracted census urban areas to {ua_dir}")
    return ua_shp


# -- Step 3: Find LCZ 6 patches ----------------------------------------------


def find_lcz6_patches(lcz_path: Path, min_patch_pixels: int = 100) -> list:
    """Find contiguous LCZ 6 patches in CONUS.

    Returns list of {lon, lat, pixels} dicts.
    Uses scipy.ndimage vectorized operations for speed (195K+ patches).
    """
    log.info("Reading LCZ map and finding LCZ 6 patches in CONUS...")

    with rasterio.open(lcz_path) as src:
        window = from_bounds(CONUS_WEST, CONUS_SOUTH, CONUS_EAST, CONUS_NORTH,
                             src.transform)
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)

    lcz6_mask = (data == LCZ_OPEN_LOWRISE)
    lcz6_count = lcz6_mask.sum()
    log.info(f"Found {lcz6_count:,} LCZ 6 pixels in CONUS ({lcz6_count * 0.01:.0f} km2)")

    from scipy import ndimage
    labeled, num_features = ndimage.label(lcz6_mask)
    log.info(f"Found {num_features:,} connected LCZ 6 patches")

    # Vectorized: compute patch sizes, centroids in one pass
    label_ids = np.arange(1, num_features + 1)
    patch_sizes = ndimage.sum(np.ones_like(labeled), labeled, label_ids).astype(int)

    # Filter by minimum size first
    big_mask = patch_sizes >= min_patch_pixels
    big_labels = label_ids[big_mask]
    big_sizes = patch_sizes[big_mask]
    log.info(f"  {len(big_labels)} patches with >= {min_patch_pixels} pixels "
             f"(skipped {num_features - len(big_labels):,} small patches)")

    # Compute centroids only for big patches
    centroids_row = ndimage.mean(
        np.arange(labeled.shape[0]).reshape(-1, 1) * np.ones(labeled.shape[1]),
        labeled, big_labels
    )
    centroids_col = ndimage.mean(
        np.ones(labeled.shape[0]).reshape(-1, 1) * np.arange(labeled.shape[1]),
        labeled, big_labels
    )

    patches = []
    for i in range(len(big_labels)):
        lon, lat = rasterio.transform.xy(win_transform, centroids_row[i], centroids_col[i])
        patches.append({"lon": lon, "lat": lat, "pixels": int(big_sizes[i])})

    log.info(f"Found {len(patches)} patches with >= {min_patch_pixels} pixels")
    return patches


# -- Step 4: Assign patches to named urban areas -----------------------------


def assign_patches_to_cities(patches: list, ua_shp: Path) -> dict:
    """Assign LCZ 6 patches to Census urban areas using spatial join."""
    if gpd is None:
        raise ImportError("geopandas required: pip install geopandas")

    log.info("Loading Census urban areas and assigning patches...")
    ua = gpd.read_file(ua_shp)
    ua = ua.cx[CONUS_WEST:CONUS_EAST, CONUS_SOUTH:CONUS_NORTH]
    log.info(f"  {len(ua)} urban areas in CONUS")

    from shapely.strtree import STRtree
    ua_geoms = ua.geometry.tolist()
    tree = STRtree(ua_geoms)

    cities = defaultdict(lambda: {
        "name": "", "lon": 0.0, "lat": 0.0,
        "population": 0, "patch_count": 0, "total_lcz6_pixels": 0,
        "patches": []
    })

    matched = 0
    for patch in patches:
        pt = Point(patch["lon"], patch["lat"])
        idx = tree.query(pt, predicate="intersects")
        if len(idx) == 0:
            continue
        i = idx[0]
        row = ua.iloc[i]
        name = row["NAME20"]
        cities[name]["name"] = name
        # Note: TIGER UAC shapefile has no POP20 column; population is
        # patched separately from Census API (fix_pop.py or plot_city_stats.py)
        cities[name]["population"] = int(row.get("ALAND20", 0))
        cities[name]["patch_count"] += 1
        cities[name]["total_lcz6_pixels"] += patch["pixels"]
        cities[name]["patches"].append(patch)
        matched += 1

    log.info(f"Matched {matched}/{len(patches)} patches to {len(cities)} urban areas")
    return dict(cities)


# -- Step 5: Select top N with COPC pre-flight --------------------------------


def select_cities_with_copc(cities: dict, n: int = 300) -> list:
    """Select top N cities by LCZ 6 coverage, filtering out those without COPC.

    For each city, picks the largest LCZ 6 patch centroid as the domain center.
    """
    ranked = sorted(cities.values(), key=lambda c: c["total_lcz6_pixels"], reverse=True)
    selected = []

    log.info(f"Checking COPC coverage for top candidate cities (target: {n})...")
    candidates_checked = 0

    for city in ranked:
        if len(selected) >= n:
            break
        if candidates_checked >= n * 3:
            log.warning(f"Checked {candidates_checked} candidates, "
                        f"only found {len(selected)} with COPC")
            break

        best_patch = max(city["patches"], key=lambda p: p["pixels"])
        lon, lat = best_patch["lon"], best_patch["lat"]

        candidates_checked += 1
        n_copc = verify_copc_coverage(lon, lat)
        if n_copc == 0:
            log.info(f"  SKIP {city['name']}: no COPC at ({lon:.4f}, {lat:.4f})")
            continue

        selected.append({
            "id": len(selected) + 1,
            "name": city["name"],
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "lcz6_pixels": city["total_lcz6_pixels"],
            "population": city["population"],
            "copc_items": n_copc,
        })

        if len(selected) % 50 == 0:
            log.info(f"  Selected {len(selected)}/{n} ({candidates_checked} checked)")

    log.info(f"Selected {len(selected)} cities with COPC ({candidates_checked} checked)")
    return selected


def run_select(output_dir: Path, n_cities: int = 300):
    """Run the full city selection pipeline (steps 1-5)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lcz_path = download_lcz(output_dir)
    ua_shp = download_census_urban_areas(output_dir)
    patches = find_lcz6_patches(lcz_path, min_patch_pixels=100)
    cities = assign_patches_to_cities(patches, ua_shp)
    selected = select_cities_with_copc(cities, n=n_cities)

    json_path = output_dir / "cities.json"
    with open(json_path, "w") as f:
        json.dump(selected, f, indent=2)
    log.info(f"Saved {len(selected)} cities to {json_path}")

    print(f"\n{'ID':>4} {'City':<40} {'Lon':>10} {'Lat':>8} {'LCZ6 px':>8} {'COPC':>5}")
    print("-" * 80)
    for c in selected[:20]:
        print(f"{c['id']:>4} {c['name']:<40} {c['lon']:>10.4f} "
              f"{c['lat']:>8.4f} {c['lcz6_pixels']:>8} {c['copc_items']:>5}")
    if len(selected) > 20:
        print(f"  ... and {len(selected) - 20} more")

    return selected


# -- Step 6: Download 3DEP data -----------------------------------------------


def get_utm_epsg(lon: float, lat: float) -> int:
    """Get UTM EPSG code for a given lon/lat."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def get_copc_items(bbox_lonlat: tuple) -> list:
    """Query Planetary Computer STAC for COPC items covering bbox."""
    if pystac_client is None:
        raise ImportError("pystac-client required")
    catalog = pystac_client.Client.open(
        STAC_API, modifier=planetary_computer.sign_inplace
    )
    search = catalog.search(collections=[COLLECTION], bbox=bbox_lonlat, limit=200)
    items = list(search.items())
    log.info(f"  Found {len(items)} COPC items")
    return items


def rasterize_copc(items: list, bbox_utm: tuple, epsg: int,
                   classes: list, agg: str = "max",
                   resolution: float = 1.0) -> np.ndarray:
    """Read COPC items, filter to classes, rasterize to grid.

    Args:
        items: PYSTAC items with COPC assets
        bbox_utm: (xmin, ymin, xmax, ymax) in UTM
        epsg: UTM EPSG code
        classes: list of ASPRS class codes to include
        agg: "max" or "mean" aggregation
        resolution: output pixel size in meters

    Returns:
        2D numpy array (rows, cols), NaN where no data
    """
    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution))
    rows = int(math.ceil((ymax - ymin) / resolution))

    if agg == "max":
        grid = np.full((rows, cols), np.nan, dtype=np.float32)
    else:
        grid_sum = np.zeros((rows, cols), dtype=np.float64)
        grid_cnt = np.zeros((rows, cols), dtype=np.int32)

    # PDAL range filter needs "Classification[v:v]" syntax per value
    class_str = ",".join(f"Classification[{c}:{c}]" for c in classes)

    for i, item in enumerate(items):
        copc_url = item.assets["data"].href
        log.info(f"    Processing COPC {i+1}/{len(items)}")

        pipeline_json = json.dumps({
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": copc_url,
                    "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])",
                    "override_srs": f"EPSG:{epsg}",
                },
                {
                    "type": "filters.range",
                    "limits": class_str
                },
            ]
        })

        try:
            p = pdal.Pipeline(pipeline_json)
            p.execute()
            if p.arrays is None or len(p.arrays) == 0 or len(p.arrays[0]) == 0:
                continue
            arr = p.arrays[0]
        except Exception as e:
            raise RuntimeError(f"COPC read failed: {copc_url}") from e

        x, y, z = arr["X"], arr["Y"], arr["Z"]
        col_idx = ((x - xmin) / resolution).astype(np.int32)
        row_idx = ((ymax - y) / resolution).astype(np.int32)

        valid = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
        col_idx, row_idx, z = col_idx[valid], row_idx[valid], z[valid]

        if len(z) == 0:
            continue

        if agg == "max":
            for r, c, zv in zip(row_idx, col_idx, z):
                if np.isnan(grid[r, c]) or zv > grid[r, c]:
                    grid[r, c] = zv
        else:
            np.add.at(grid_sum, (row_idx, col_idx), z)
            np.add.at(grid_cnt, (row_idx, col_idx), 1)

    if agg == "max":
        return grid
    else:
        mask = grid_cnt > 0
        result = np.full((rows, cols), np.nan, dtype=np.float32)
        result[mask] = (grid_sum[mask] / grid_cnt[mask]).astype(np.float32)
        return result


def rasterize_copc_all(items: list, bbox_utm: tuple, epsg: int,
                       agg: str = "max", resolution: float = 1.0,
                       exclude_classes: list = None) -> np.ndarray:
    """Read ALL COPC points (no class filter), optionally excluding some classes.

    Used for full DSM when vegetation class codes are non-standard.
    """
    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution))
    rows = int(math.ceil((ymax - ymin) / resolution))

    if agg == "max":
        grid = np.full((rows, cols), np.nan, dtype=np.float32)
    else:
        grid_sum = np.zeros((rows, cols), dtype=np.float64)
        grid_cnt = np.zeros((rows, cols), dtype=np.int32)

    for i, item in enumerate(items):
        copc_url = item.assets["data"].href
        log.info(f"    Processing COPC {i+1}/{len(items)} (all classes)")

        pipeline_stages = [
            {
                "type": "readers.copc",
                "filename": copc_url,
                "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])",
                "override_srs": f"EPSG:{epsg}",
            },
        ]
        # Exclude noise classes if requested
        if exclude_classes:
            exclude_str = ",".join(
                f"Classification![{c}:{c}]" for c in exclude_classes
            )
            pipeline_stages.append({
                "type": "filters.range",
                "limits": exclude_str
            })

        pipeline_json = json.dumps({"pipeline": pipeline_stages})

        try:
            p = pdal.Pipeline(pipeline_json)
            p.execute()
            if p.arrays is None or len(p.arrays) == 0 or len(p.arrays[0]) == 0:
                continue
            arr = p.arrays[0]
        except Exception as e:
            raise RuntimeError(f"COPC read failed: {copc_url}") from e

        x, y, z = arr["X"], arr["Y"], arr["Z"]
        col_idx = ((x - xmin) / resolution).astype(np.int32)
        row_idx = ((ymax - y) / resolution).astype(np.int32)

        valid = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
        col_idx, row_idx, z = col_idx[valid], row_idx[valid], z[valid]

        if len(z) == 0:
            continue

        if agg == "max":
            for r, c, zv in zip(row_idx, col_idx, z):
                if np.isnan(grid[r, c]) or zv > grid[r, c]:
                    grid[r, c] = zv
        else:
            np.add.at(grid_sum, (row_idx, col_idx), z)
            np.add.at(grid_cnt, (row_idx, col_idx), 1)

    if agg == "max":
        return grid
    else:
        mask = grid_cnt > 0
        result = np.full((rows, cols), np.nan, dtype=np.float32)
        result[mask] = (grid_sum[mask] / grid_cnt[mask]).astype(np.float32)
        return result


def _build_cdsm_from_returns(items: list, bbox_utm: tuple, epsg: int,
                             dem: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """Build CDSM using multi-return points (vegetation) vs single-return (buildings).

    When standard vegetation classes (3/4/5) are absent, use return number:
    - Multi-return points above ground = tree canopy
    - Single-return points above ground = likely buildings
    """
    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution))
    rows = int(math.ceil((ymax - ymin) / resolution))
    grid = np.full((rows, cols), np.nan, dtype=np.float32)

    for i, item in enumerate(items):
        copc_url = item.assets["data"].href
        log.info(f"    Processing COPC {i+1}/{len(items)} (multi-return canopy)")

        pipeline_json = json.dumps({
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": copc_url,
                    "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])",
                    "override_srs": f"EPSG:{epsg}",
                },
                {
                    # Exclude ground (2) and noise (7)
                    "type": "filters.range",
                    "limits": "Classification![2:2],Classification![7:7],"
                              "NumberOfReturns[2:]"
                },
            ]
        })

        try:
            p = pdal.Pipeline(pipeline_json)
            p.execute()
            if p.arrays is None or len(p.arrays) == 0 or len(p.arrays[0]) == 0:
                continue
            arr = p.arrays[0]
        except Exception as e:
            raise RuntimeError(f"COPC read failed: {copc_url}") from e

        x, y, z = arr["X"], arr["Y"], arr["Z"]
        col_idx = ((x - xmin) / resolution).astype(np.int32)
        row_idx = ((ymax - y) / resolution).astype(np.int32)

        valid = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
        col_idx, row_idx, z = col_idx[valid], row_idx[valid], z[valid]

        if len(z) == 0:
            continue

        for r, c, zv in zip(row_idx, col_idx, z):
            if np.isnan(grid[r, c]) or zv > grid[r, c]:
                grid[r, c] = zv

    # Convert to height above ground
    cdsm = np.nan_to_num(grid, nan=0.0) - np.nan_to_num(dem, nan=0.0)
    return cdsm


def gap_fill(data: np.ndarray, max_dist: int = GAP_FILL_DISTANCE) -> np.ndarray:
    """Fill NaN gaps using rasterio's fillnodata."""
    from rasterio.fill import fillnodata
    if data.ndim != 2 or data.size == 0:
        raise ValueError("gap_fill requires a nonempty two-dimensional array")
    if np.isinf(data).any():
        raise ValueError("gap_fill input contains infinite values")
    mask = ~np.isnan(data)
    if not mask.any():
        raise ValueError("cannot fill a raster containing only nodata")
    if mask.all():
        return data
    return fillnodata(data.copy(), mask.astype(np.uint8), max_search_distance=max_dist)


def save_tif(data: np.ndarray, path: Path, bbox_utm: tuple, epsg: int,
             resolution: float = 1.0, nodata: float = -9999.0):
    """Save 2D array as GeoTIFF."""
    if data.ndim != 2 or data.size == 0:
        raise ValueError("GeoTIFF data must be a nonempty two-dimensional array")
    if np.isinf(data).any() or not np.isfinite(data).any():
        raise ValueError("GeoTIFF data must contain finite values and no infinities")
    xmin, ymin, xmax, ymax = bbox_utm
    rows, cols = data.shape
    transform = rasterio.transform.from_bounds(
        xmin, ymax - rows * resolution, xmin + cols * resolution, ymax, cols, rows
    )
    out = data.copy()
    out[np.isnan(out)] = nodata

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=rows, width=cols, count=1,
        dtype="float32", crs=CRS.from_epsg(epsg),
        transform=transform, nodata=nodata,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write(out, 1)
    log.info(f"  Saved {path} ({rows}x{cols})")


def _latlon_to_quadkey(lat: float, lon: float, level: int = 9) -> str:
    """Convert lat/lon to Bing Maps quadkey at given zoom level."""
    x = (lon + 180.0) / 360.0
    sin_lat = math.sin(lat * math.pi / 180.0)
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    map_size = 1 << level
    pixel_x = int(min(max(x * map_size, 0), map_size - 1))
    pixel_y = int(min(max(y * map_size, 0), map_size - 1))
    quadkey = ""
    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (pixel_x & mask) != 0:
            digit += 1
        if (pixel_y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


def get_building_mask(lon: float, lat: float, bbox_utm: tuple, epsg: int,
                      resolution: float = 1.0) -> np.ndarray:
    """Download MS Building Footprints and rasterize to 1m binary mask.

    Uses Microsoft Building Footprints from Planetary Computer (999M+ buildings,
    ML-detected from Bing Maps imagery, 99.3% precision in US).

    Returns:
        2D boolean array (rows, cols), True where buildings exist.
    """
    import geopandas as gpd
    from rasterio.features import rasterize as rio_rasterize

    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution))
    rows = int(math.ceil((ymax - ymin) / resolution))
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    # Get signed STAC asset for MS Buildings
    catalog = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[MS_BUILDINGS_COLLECTION],
        bbox=[lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02]
    )
    items = list(search.items())
    if not items:
        log.warning("  No MS Building Footprints found, returning empty mask")
        return np.zeros((rows, cols), dtype=bool)

    item = max(items, key=lambda candidate: candidate.properties.get("datetime", ""))
    asset = item.assets["data"]
    storage_options = asset.extra_fields.get("table:storage_options", {})

    # Compute quadkey(s) covering our bbox
    # Check corners to handle quadkey boundaries
    inv_t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    corners = [
        inv_t.transform(xmin, ymin), inv_t.transform(xmax, ymin),
        inv_t.transform(xmin, ymax), inv_t.transform(xmax, ymax),
    ]
    quadkeys = set()
    for clon, clat in corners:
        quadkeys.add(_latlon_to_quadkey(clat, clon, 9))
    quadkeys.add(_latlon_to_quadkey(lat, lon, 9))

    log.info(f"  MS Building Footprints: querying {len(quadkeys)} quadkey(s)")

    # Read building footprints from each quadkey partition
    all_gdfs = []
    successful_queries = 0
    bbox_lonlat = box(
        *inv_t.transform(xmin, ymin), *inv_t.transform(xmax, ymax)
    )
    for qk in quadkeys:
        parquet_path = f"{asset.href}/quadkey={qk}"
        try:
            gdf = gpd.read_parquet(parquet_path, storage_options=storage_options)
            successful_queries += 1
            gdf_clip = gdf[gdf.intersects(bbox_lonlat)]
            if len(gdf_clip) > 0:
                all_gdfs.append(gdf_clip)
            log.info(f"    quadkey {qk}: {len(gdf_clip)} buildings")
        except Exception as e:
            log.warning(f"    quadkey {qk} failed: {e}")

    if successful_queries == 0:
        raise RuntimeError("all MS Building Footprints partition queries failed")
    if not all_gdfs:
        log.warning("  No buildings found in MS footprints")
        return np.zeros((rows, cols), dtype=bool)

    buildings = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    buildings = buildings.set_crs("EPSG:4326")
    buildings = buildings.to_crs(f"EPSG:{epsg}")

    log.info(f"  Rasterizing {len(buildings)} building footprints to {rows}x{cols} mask")

    # Rasterize to binary mask
    shapes = [(geom, 1) for geom in buildings.geometry if geom is not None]
    if not shapes:
        return np.zeros((rows, cols), dtype=bool)

    mask = rio_rasterize(
        shapes, out_shape=(rows, cols), transform=transform,
        fill=0, dtype=np.uint8
    )
    return mask.astype(bool)


def rasterize_copc_masked(items: list, bbox_utm: tuple, epsg: int,
                          building_mask: np.ndarray, target: str,
                          resolution: float = 1.0) -> np.ndarray:
    """Read ALL COPC points, split by building mask, rasterize max Z.

    Args:
        target: "building" keeps points IN building mask,
                "vegetation" keeps points OUTSIDE building mask.
    """
    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution))
    rows = int(math.ceil((ymax - ymin) / resolution))
    grid = np.full((rows, cols), np.nan, dtype=np.float32)

    for i, item in enumerate(items):
        copc_url = item.assets["data"].href
        log.info(f"    Processing COPC {i+1}/{len(items)} ({target})")

        # Read all points except noise (class 7)
        pipeline_json = json.dumps({
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": copc_url,
                    "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])",
                    "override_srs": f"EPSG:{epsg}",
                },
                {
                    "type": "filters.range",
                    "limits": "Classification![7:7]"
                },
            ]
        })

        try:
            p = pdal.Pipeline(pipeline_json)
            p.execute()
            if p.arrays is None or len(p.arrays) == 0 or len(p.arrays[0]) == 0:
                continue
            arr = p.arrays[0]
        except Exception as e:
            raise RuntimeError(f"COPC read failed: {copc_url}") from e

        x, y, z = arr["X"], arr["Y"], arr["Z"]
        col_idx = ((x - xmin) / resolution).astype(np.int32)
        row_idx = ((ymax - y) / resolution).astype(np.int32)

        valid = (col_idx >= 0) & (col_idx < cols) & (row_idx >= 0) & (row_idx < rows)
        col_idx, row_idx, z = col_idx[valid], row_idx[valid], z[valid]

        if len(z) == 0:
            continue

        # Apply building mask filter
        in_building = building_mask[row_idx, col_idx]
        if target == "building":
            keep = in_building
        else:  # vegetation
            keep = ~in_building
        col_idx, row_idx, z = col_idx[keep], row_idx[keep], z[keep]

        if len(z) == 0:
            continue

        for r, c, zv in zip(row_idx, col_idx, z):
            if np.isnan(grid[r, c]) or zv > grid[r, c]:
                grid[r, c] = zv

    return grid


def get_landcover(lon: float, lat: float, bbox_utm: tuple, epsg: int,
                  resolution: float = 1.0) -> np.ndarray:
    """Download ESA WorldCover 2021 and reproject to 1m UTM grid.

    Returns uint8 array with ESA WorldCover class codes (10-100).
    Nearest-neighbor resampling preserves categorical values.
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds

    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution))
    rows = int(math.ceil((ymax - ymin) / resolution))
    dst_transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    # Convert UTM bbox to lon/lat for STAC query
    inv_t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = inv_t.transform(xmin, ymin)
    lon_max, lat_max = inv_t.transform(xmax, ymax)

    catalog = pystac_client.Client.open(STAC_API, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[ESA_WORLDCOVER_COLLECTION],
        bbox=[lon_min, lat_min, lon_max, lat_max],
        query={"esa_worldcover:product_version": {"eq": "V200"}},
    )
    items = list(search.items())

    # Fall back to any version if V200 not found
    if not items:
        search = catalog.search(
            collections=[ESA_WORLDCOVER_COLLECTION],
            bbox=[lon_min, lat_min, lon_max, lat_max],
        )
        items = list(search.items())

    if not items:
        raise RuntimeError("no ESA WorldCover data found for the requested area")

    # Use first (newest) item
    asset = items[0].assets["map"]
    log.info(f"  ESA WorldCover: {items[0].id}")

    with rasterio.open(asset.href) as src:
        # Read window covering our bbox (in WGS84)
        buf = 0.002  # small buffer in degrees
        win = from_bounds(
            lon_min - buf, lat_min - buf, lon_max + buf, lat_max + buf,
            src.transform
        )
        src_data = src.read(1, window=win)
        src_transform = rasterio.windows.transform(win, src.transform)

        # Reproject to UTM 1m grid (nearest neighbor for categorical data)
        dst_data = np.zeros((rows, cols), dtype=np.uint8)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=CRS.from_epsg(epsg),
            resampling=Resampling.nearest,
        )

    classes, counts = np.unique(dst_data, return_counts=True)
    class_summary = ", ".join(
        f"{ESA_WC_CLASSES.get(c, '?')}={n/dst_data.size*100:.0f}%"
        for c, n in zip(classes, counts) if c > 0
    )
    log.info(f"  Land cover (ESA): {class_summary}")

    # Reclassify to UTherm codes (1-7)
    utherm_lc = np.zeros_like(dst_data)
    for esa_code, utherm_code in ESA_TO_UTHERM.items():
        utherm_lc[dst_data == esa_code] = utherm_code
    # Default unmapped pixels to asphalt (1)
    utherm_lc[utherm_lc == 0] = 1

    utherm_names = {1: "asphalt", 5: "grass", 6: "bare_soil", 7: "water"}
    ut_classes, ut_counts = np.unique(utherm_lc, return_counts=True)
    ut_summary = ", ".join(
        f"{utherm_names.get(c, '?')}={n/utherm_lc.size*100:.0f}%"
        for c, n in zip(ut_classes, ut_counts) if c > 0
    )
    log.info(f"  Land cover (UTherm): {ut_summary}")

    return utherm_lc


def raster_is_valid(path: Path, kind: str = "continuous") -> bool:
    """Check whether a previously written study raster is safe to reuse."""
    if not path.is_file():
        return False
    expected_size = int(round(PATCH_SIZE_M / RESOLUTION))
    try:
        with rasterio.open(path) as src:
            if (
                src.count != 1
                or (src.height, src.width) != (expected_size, expected_size)
                or src.crs is None
                or not np.isclose(abs(src.transform.a), RESOLUTION)
                or not np.isclose(abs(src.transform.e), RESOLUTION)
            ):
                return False
            data = src.read(1, masked=True).filled(np.nan).astype(float)
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return False
    finite = np.isfinite(data)
    if not finite.any() or np.isinf(data).any():
        return False
    values = data[finite]
    if kind == "cdsm" and (values.min() < 0.0 or values.max() > 50.0):
        return False
    if kind == "landcover":
        integer_values = values.astype(int)
        if not np.allclose(values, integer_values) or not set(np.unique(integer_values)) <= {1, 5, 6, 7}:
            return False
    return True


def download_city(city: dict, output_dir: Path, aoi_id: int = None):
    """Download DEM, Building_DSM, and CDSM for one 1.2km x 1.2km city patch.

    If aoi_id is given, output goes to <output_dir>/<city_id>/aoi_<aoi_id>/,
    letting a city host multiple AOIs in parallel subdirs. The city dict's
    lon/lat fields should already be the AOI-specific center.
    """
    cid = city["id"]
    lon, lat = city["lon"], city["lat"]
    name = city["name"]

    if aoi_id is not None:
        city_dir = output_dir / str(cid) / f"aoi_{aoi_id}"
    else:
        city_dir = output_dir / str(cid)
    dem_path = city_dir / "DEM.tif"
    bdsm_path = city_dir / "Building_DSM.tif"
    cdsm_path = city_dir / "CDSM.tif"
    lc_path = city_dir / "landcover.tif"

    existing_valid = {
        dem_path: raster_is_valid(dem_path),
        bdsm_path: raster_is_valid(bdsm_path),
        cdsm_path: raster_is_valid(cdsm_path, "cdsm"),
        lc_path: raster_is_valid(lc_path, "landcover"),
    }
    if all(existing_valid.values()):
        log.info(f"City {cid} ({name}) already complete, skipping")
        return

    log.info(f"=== City {cid}: {name} ({lon:.4f}, {lat:.4f}) ===")

    epsg = get_utm_epsg(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = transformer.transform(lon, lat)
    half = PATCH_SIZE_M / 2
    bbox_utm = (cx - half, cy - half, cx + half, cy + half)

    inv_transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    buf = 100
    lon_min, lat_min = inv_transformer.transform(cx - half - buf, cy - half - buf)
    lon_max, lat_max = inv_transformer.transform(cx + half + buf, cy + half + buf)
    bbox_lonlat = (lon_min, lat_min, lon_max, lat_max)

    items = get_copc_items(bbox_lonlat)
    if not items:
        _save_metadata(city, city_dir, epsg, bbox_utm, has_copc=False)
        raise RuntimeError(f"no COPC data for city {cid} ({name})")

    # Step 1: DEM from ground points (class 2) — universal across all datasets
    if not existing_valid[dem_path]:
        log.info("  Building DEM (ground, class 2, mean)...")
        dem = rasterize_copc(items, bbox_utm, epsg, CLASS_GROUND, agg="mean")
        dem = gap_fill(dem)
        save_tif(dem, dem_path, bbox_utm, epsg)

    # Step 2: Get MS Building Footprints mask (ML-detected, classification-independent)
    if not existing_valid[bdsm_path] or not existing_valid[cdsm_path]:
        log.info("  Downloading MS Building Footprints...")
        building_mask = get_building_mask(lon, lat, bbox_utm, epsg)
        n_bldg_px = building_mask.sum()
        log.info(f"  Building mask: {n_bldg_px:,} pixels "
                 f"({n_bldg_px / building_mask.size * 100:.1f}%)")

    # Step 3: Building_DSM = all COPC points WITHIN building footprints, max Z
    if not existing_valid[bdsm_path]:
        log.info("  Building Building_DSM (COPC points in MS footprints, max Z)...")
        bdsm = rasterize_copc_masked(items, bbox_utm, epsg, building_mask,
                                     target="building")
        # Also fill with DEM where no building points but mask says building
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            dem[dem == src.nodata] = np.nan
        # Fill building gaps with DEM (some footprints may lack COPC points)
        bdsm_filled = np.where(np.isnan(bdsm) & building_mask, dem, bdsm)
        # Fill non-building areas with DEM
        bdsm_filled = np.where(np.isnan(bdsm_filled), dem, bdsm_filled)
        bdsm_filled = gap_fill(bdsm_filled)
        # Remove COPC noise: cap building height at 200m above DEM
        # Re-read gap-filled DEM (dem var may have NaN from nodata)
        with rasterio.open(dem_path) as src:
            dem_clean = src.read(1).astype(np.float32)
            dem_clean[dem_clean == src.nodata] = 0.0
        bldg_height = bdsm_filled - dem_clean
        noise = bldg_height > 200.0
        if noise.any():
            log.warning(f"  Removing {noise.sum()} noise pixels > 200m above DEM")
            bdsm_filled = np.where(noise, dem_clean, bdsm_filled)
        save_tif(bdsm_filled, bdsm_path, bbox_utm, epsg)

    # Step 4: CDSM = all COPC points OUTSIDE building footprints, max Z - DEM
    if not existing_valid[cdsm_path]:
        log.info("  Building CDSM (COPC points outside MS footprints, max Z - DEM)...")
        veg_dsm = rasterize_copc_masked(items, bbox_utm, epsg, building_mask,
                                        target="vegetation")
        if not np.isfinite(veg_dsm).any():
            raise RuntimeError("COPC vegetation rasterization returned no usable points")

        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            dem[dem == src.nodata] = np.nan

        cdsm = veg_dsm - dem
        cdsm = np.nan_to_num(cdsm, nan=0.0)
        cdsm[cdsm < 2.0] = 0.0   # Remove low vegetation (< 2m) — handled by landcover
        cdsm[cdsm > 50.0] = 50.0  # Cap noise

        n_tree_px = (cdsm > 0.0).sum()
        pct = n_tree_px / cdsm.size * 100
        log.info(f"  CDSM: {n_tree_px:,} tree pixels ({pct:.1f}%), "
                 f"max height={cdsm.max():.1f}m")
        save_tif(cdsm, cdsm_path, bbox_utm, epsg)

    # Step 5: ESA WorldCover land cover (10m → 1m nearest-neighbor)
    if not existing_valid[lc_path]:
        log.info("  Downloading ESA WorldCover 2021...")
        lc = get_landcover(lon, lat, bbox_utm, epsg)
        save_tif(lc.astype(np.float32), lc_path, bbox_utm, epsg, nodata=0)

    _save_metadata(city, city_dir, epsg, bbox_utm, has_copc=True)
    log.info(f"  City {cid} ({name}) complete")


def _save_metadata(city: dict, city_dir: Path, epsg: int, bbox_utm: tuple,
                   has_copc: bool):
    """Save city metadata JSON."""
    meta = {
        **city,
        "epsg": epsg,
        "bbox_utm": list(bbox_utm),
        "patch_size_m": PATCH_SIZE_M,
        "analysis_size_m": ANALYSIS_SIZE_M,
        "resolution_m": RESOLUTION,
        "has_copc": has_copc,
    }
    meta_path = city_dir / "metadata.json"
    city_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _download_city_wrapper(args):
    """Wrapper for multiprocessing Pool."""
    city, data_dir = args
    try:
        download_city(city, data_dir)
        return city["id"], True, None
    except Exception as e:
        return city["id"], False, str(e)


def run_download(output_dir: Path, city_id: int = None, n_workers: int = 4):
    """Download 3DEP data for cities listed in cities.json."""
    json_path = output_dir / "cities.json"
    if not json_path.exists():
        raise FileNotFoundError("cities.json not found; run --step select first")

    with open(json_path) as f:
        cities = json.load(f)

    if city_id is not None:
        cities = [c for c in cities if c["id"] == city_id]
        if not cities:
            raise ValueError(f"city ID {city_id} not found in cities.json")

    # Store city data in subdirectory
    data_dir = output_dir / "300_cities_data"
    data_dir.mkdir(exist_ok=True)

    log.info(f"Downloading data for {len(cities)} cities to {data_dir} "
             f"with {n_workers} workers...")
    failures = []

    if n_workers <= 1 or city_id is not None:
        # Sequential mode
        for i, city in enumerate(cities):
            log.info(f"[{i+1}/{len(cities)}] City {city['id']}: {city['name']}")
            try:
                download_city(city, data_dir)
            except Exception as e:
                log.error(f"  FAILED: {e}")
                failures.append((city["id"], str(e)))
    else:
        # Parallel mode
        from multiprocessing.pool import ThreadPool
        args_list = [(city, data_dir) for city in cities]
        with ThreadPool(n_workers) as pool:
            for i, (cid, ok, err) in enumerate(
                pool.imap_unordered(_download_city_wrapper, args_list)
            ):
                status = "OK" if ok else f"FAILED: {err}"
                log.info(f"  [{i+1}/{len(cities)}] City {cid}: {status}")
                if not ok:
                    failures.append((cid, err))
    if failures:
        log.error(f"Failed to prepare {len(failures)}/{len(cities)} cities")
    return failures


# -- Main ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Select and download data for Nature Cities experiment"
    )
    parser.add_argument("--step", choices=["select", "download"], required=True,
                        help="'select' to pick cities, 'download' to get 3DEP data")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--n-cities", type=int, default=300,
                        help="Number of cities to select (default: 300)")
    parser.add_argument("--city", type=int, default=None,
                        help="Download a single city by ID")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    if args.step == "select":
        run_select(args.output_dir, args.n_cities)
    elif args.step == "download":
        failures = run_download(args.output_dir, args.city, args.workers)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
