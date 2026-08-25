#!/usr/bin/env python
"""prepare_usgs_city.py — data-prep for cities ABSENT from the Planetary Computer
3dep-lidar-copc catalog, using USGS 3DEP LAZ tiles from The National Map (TNM).

Why TNM-LAZ and not EPT: the hobuinc usgs-lidar entwine index is stale — it lacks
the recent (2018-2024) projects that actually cover these interior-west cities (its
older resources overlap only the cube-padded bounds, not the city). TNM serves the
current per-tile LAZ on rockyweb. Those tiles are plain LAZ (not COPC), so we
download the bbox tiles once to a local cache and read them with readers.las.

Everything except the point source mirrors prepare_nature_cities.download_city
(DEM ground-mean, Building_DSM in MS-footprints max-Z, CDSM outside-footprints
max-Z minus DEM, ESA landcover, gap-fill).

Usage:
  python prepare_usgs_city.py --city 304 --output-dir /path/to/data \
         --cities-json /path/to/data/cities.json
"""
import argparse, json, math, os, re, sys, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_nature_cities as P

pdal = P.pdal

TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
log = P.log


# ── TNM discovery + tile download ────────────────────────────────────────────

def _project_year(name):
    """crude recency score: 4-digit year or D<yy>/B<yy> code in the project name."""
    yrs = [int(y) for y in re.findall(r"(?:19|20)\d{2}", name)]
    codes = [2000 + int(c) for c in re.findall(r"[BDC](\d{2})\b", name)]
    cand = yrs + codes
    return max(cand) if cand else 0


def get_tnm_laz_urls(bbox_lonlat, max_items=400):
    """Return LAZ tile URLs covering bbox, restricted to the single newest project."""
    lo0, la0, lo1, la1 = bbox_lonlat
    q = urllib.parse.urlencode({"datasets": "Lidar Point Cloud (LPC)",
                                "bbox": f"{lo0},{la0},{lo1},{la1}",
                                "outputFormat": "JSON", "max": max_items})
    d = json.load(urllib.request.urlopen(f"{TNM}?{q}", timeout=60))
    by_proj = {}
    for it in d.get("items", []):
        url = it.get("downloadURL") or ""
        if not url.lower().endswith(".laz"):
            continue
        title = it.get("title", "")
        proj = title.replace("USGS Lidar Point Cloud ", "").rsplit(" ", 1)[0]
        by_proj.setdefault(proj, []).append(url)
    if not by_proj:
        return []
    # prefer the newest project (most recent year), tie-break by tile count
    best = max(by_proj, key=lambda p: (_project_year(p), len(by_proj[p])))
    log.info(f"  TNM: {len(by_proj)} projects; using '{best}' (year~{_project_year(best)}, {len(by_proj[best])} tiles)")
    return by_proj[best]


def download_tiles(urls, dest):
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, u in enumerate(urls):
        fn = dest / os.path.basename(urllib.parse.urlparse(u).path)
        if not fn.exists() or fn.stat().st_size == 0:
            log.info(f"    download tile {i+1}/{len(urls)}: {fn.name}")
            partial = fn.with_suffix(fn.suffix + ".part")
            try:
                urllib.request.urlretrieve(u, partial)
                if partial.stat().st_size == 0:
                    raise RuntimeError("downloaded file is empty")
                partial.replace(fn)
            except Exception as e:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"tile download failed: {u}") from e
        paths.append(str(fn))
    return paths


def _las_stages(laz, bbox_utm, epsg):
    """readers.las (local) -> reproject to city UTM -> crop to bbox."""
    xmin, ymin, xmax, ymax = bbox_utm
    return [
        {"type": "readers.las", "filename": laz},
        {"type": "filters.reprojection", "out_srs": f"EPSG:{epsg}"},
        {"type": "filters.crop", "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])"},
    ]


# ── rasterizers (mirror rasterize_copc / rasterize_copc_masked) ───────────────

def rasterize_las(laz_paths, bbox_utm, epsg, classes, agg="max", resolution=1.0):
    if pdal is None:
        raise ImportError("PDAL is required for USGS LAZ preprocessing")
    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution)); rows = int(math.ceil((ymax - ymin) / resolution))
    if agg == "max":
        grid = np.full((rows, cols), np.nan, dtype=np.float32)
    else:
        gsum = np.zeros((rows, cols)); gcnt = np.zeros((rows, cols), dtype=np.int32)
    cstr = ",".join(f"Classification[{c}:{c}]" for c in classes)
    for i, laz in enumerate(laz_paths):
        st = _las_stages(laz, bbox_utm, epsg) + [{"type": "filters.range", "limits": cstr}]
        try:
            p = pdal.Pipeline(json.dumps({"pipeline": st})); p.execute()
            if not p.arrays or len(p.arrays[0]) == 0: continue
            a = p.arrays[0]
        except Exception as e:
            raise RuntimeError(f"LAS read failed: {os.path.basename(laz)}") from e
        x, y, z = a["X"], a["Y"], a["Z"]
        ci = ((x - xmin) / resolution).astype(np.int32); ri = ((ymax - y) / resolution).astype(np.int32)
        v = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows); ci, ri, z = ci[v], ri[v], z[v]
        if len(z) == 0: continue
        if agg == "max":
            for r, c, zv in zip(ri, ci, z):
                if np.isnan(grid[r, c]) or zv > grid[r, c]: grid[r, c] = zv
        else:
            np.add.at(gsum, (ri, ci), z); np.add.at(gcnt, (ri, ci), 1)
    if agg == "max":
        return grid
    res = np.full((rows, cols), np.nan, dtype=np.float32); m = gcnt > 0
    res[m] = (gsum[m] / gcnt[m]).astype(np.float32); return res


def rasterize_las_masked(laz_paths, bbox_utm, epsg, bmask, target, resolution=1.0):
    if pdal is None:
        raise ImportError("PDAL is required for USGS LAZ preprocessing")
    xmin, ymin, xmax, ymax = bbox_utm
    cols = int(math.ceil((xmax - xmin) / resolution)); rows = int(math.ceil((ymax - ymin) / resolution))
    grid = np.full((rows, cols), np.nan, dtype=np.float32)
    for laz in laz_paths:
        st = _las_stages(laz, bbox_utm, epsg) + [{"type": "filters.range", "limits": "Classification![7:7]"}]
        try:
            p = pdal.Pipeline(json.dumps({"pipeline": st})); p.execute()
            if not p.arrays or len(p.arrays[0]) == 0: continue
            a = p.arrays[0]
        except Exception as e:
            raise RuntimeError(f"LAS read failed: {os.path.basename(laz)}") from e
        x, y, z = a["X"], a["Y"], a["Z"]
        ci = ((x - xmin) / resolution).astype(np.int32); ri = ((ymax - y) / resolution).astype(np.int32)
        v = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows); ci, ri, z = ci[v], ri[v], z[v]
        if len(z) == 0: continue
        inb = bmask[ri, ci]; keep = inb if target == "building" else ~inb
        ci, ri, z = ci[keep], ri[keep], z[keep]
        for r, c, zv in zip(ri, ci, z):
            if np.isnan(grid[r, c]) or zv > grid[r, c]: grid[r, c] = zv
    return grid


# ── download_city, TNM-LAZ edition (mirrors P.download_city) ──────────────────

def download_city_usgs(city, output_dir):
    cid, lon, lat, name = city["id"], city["lon"], city["lat"], city["name"]
    city_dir = output_dir / "300_cities_data" / str(cid)
    dem_p = city_dir / "DEM.tif"; bdsm_p = city_dir / "Building_DSM.tif"
    cdsm_p = city_dir / "CDSM.tif"; lc_p = city_dir / "landcover.tif"
    existing_valid = {
        dem_p: P.raster_is_valid(dem_p),
        bdsm_p: P.raster_is_valid(bdsm_p),
        cdsm_p: P.raster_is_valid(cdsm_p, "cdsm"),
        lc_p: P.raster_is_valid(lc_p, "landcover"),
    }
    if all(existing_valid.values()):
        log.info(f"City {cid} ({name}) already complete, skipping"); return
    city_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"=== USGS/TNM City {cid}: {name} ({lon:.4f}, {lat:.4f}) ===")

    epsg = P.get_utm_epsg(lon, lat)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = tf.transform(lon, lat); half = P.PATCH_SIZE_M / 2
    bbox_utm = (cx - half, cy - half, cx + half, cy + half)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lo0, la0 = inv.transform(cx - half - 100, cy - half - 100)
    lo1, la1 = inv.transform(cx + half + 100, cy + half + 100)

    urls = get_tnm_laz_urls((lo0, la0, lo1, la1))
    if not urls:
        P._save_metadata(city, city_dir, epsg, bbox_utm, has_copc=False)
        raise RuntimeError(f"no TNM LAZ for city {cid}")
    laz = download_tiles(urls, Path(city_dir).parent / "tnm_laz" / str(cid))
    if not laz:
        raise RuntimeError(f"no TNM LAZ tiles downloaded for city {cid}")
    log.info(f"  {len(laz)} LAZ tiles ready")

    if not existing_valid[dem_p]:
        log.info("  Building DEM (ground class 2, mean)...")
        P.save_tif(P.gap_fill(rasterize_las(laz, bbox_utm, epsg, P.CLASS_GROUND, agg="mean")), dem_p, bbox_utm, epsg)

    if not existing_valid[bdsm_p] or not existing_valid[cdsm_p]:
        log.info("  Downloading MS Building Footprints...")
        bmask = P.get_building_mask(lon, lat, bbox_utm, epsg)
        log.info(f"  Building mask: {int(bmask.sum()):,} px ({bmask.sum()/bmask.size*100:.1f}%)")

    if not existing_valid[bdsm_p]:
        log.info("  Building Building_DSM (LAS pts in MS footprints, max Z)...")
        bdsm = rasterize_las_masked(laz, bbox_utm, epsg, bmask, "building")
        with rasterio.open(dem_p) as s: dem = s.read(1); dem[dem == s.nodata] = np.nan
        bdsm = P.gap_fill(np.where(np.isnan(bdsm), dem, np.where(np.isnan(bdsm) & bmask, dem, bdsm)))
        with rasterio.open(dem_p) as s: demc = s.read(1).astype(np.float32); demc[demc == s.nodata] = 0.0
        noise = (bdsm - demc) > 200.0
        if noise.any(): bdsm = np.where(noise, demc, bdsm)
        P.save_tif(bdsm, bdsm_p, bbox_utm, epsg)

    if not existing_valid[cdsm_p]:
        log.info("  Building CDSM (LAS pts outside MS footprints, max Z - DEM)...")
        veg = rasterize_las_masked(laz, bbox_utm, epsg, bmask, "vegetation")
        if not np.isfinite(veg).any():
            raise RuntimeError("TNM vegetation rasterization returned no usable points")
        with rasterio.open(dem_p) as s: dem = s.read(1); dem[dem == s.nodata] = np.nan
        cdsm = np.nan_to_num(veg - dem, nan=0.0); cdsm[cdsm < 2.0] = 0.0; cdsm[cdsm > 50.0] = 50.0
        log.info(f"  CDSM: {int((cdsm>0).sum()):,} tree px ({(cdsm>0).sum()/cdsm.size*100:.1f}%), max={cdsm.max():.1f}m")
        P.save_tif(cdsm, cdsm_p, bbox_utm, epsg)

    if not existing_valid[lc_p]:
        log.info("  Downloading ESA WorldCover 2021...")
        P.save_tif(P.get_landcover(lon, lat, bbox_utm, epsg).astype(np.float32), lc_p, bbox_utm, epsg, nodata=0)

    P._save_metadata(city, city_dir, epsg, bbox_utm, has_copc=True)
    log.info(f"  City {cid} ({name}) complete (USGS/TNM)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cities-json", type=Path, required=True)
    a = ap.parse_args()
    c = [x for x in json.load(open(a.cities_json)) if x["id"] == a.city]
    if not c:
        log.error(f"city {a.city} not in {a.cities_json}"); sys.exit(1)
    download_city_usgs(c[0], a.output_dir)


if __name__ == "__main__":
    main()
