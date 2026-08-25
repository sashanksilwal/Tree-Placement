"""Download OSM constraints and build reproducible tree-placement masks.

The placement workflow has two explicit domains:

``everywhere``
    Any canopy-free, non-building centre that is not mapped as a road surface,
    sidewalk/path, parking area, water, or swimming pool.

``street-verge``
    Centres in the per-class planting ring beside an OSM road centreline,
    supplemented by mapped sidewalk buffers.  The ring runs from the class
    carriageway half-width (``inner``) out to the class planting-strip cap
    (``outer``), so the paved surface is excluded by construction: a motorway
    contributes a 15--25 m ring, a residential street a 3.5--7 m ring.  This
    reproduces the archived ``modify_osm_streets`` definition, which planted
    *beside* every mapped highway class rather than on it.

OSM is incomplete and does not encode ownership, utilities, soil volume, or
planting permission.  These masks are therefore model eligibility masks, not a
claim that every retained location is operationally plantable.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
)

LAYER_QUERIES = {
    "highway": "way[highway]",
    # Only mapped sidewalk *geometry*.  A bare ``nwr[sidewalk]`` would match the
    # road centreline carrying a ``sidewalk=*`` attribute -- including
    # ``sidewalk=no`` -- and buffering that centreline is not a sidewalk.
    "sidewalk": "nwr[highway=footway][footway=sidewalk]",
    "building": "nwr[building]",
    "parking": "nwr[amenity=parking]",
    "residential": "nwr[landuse=residential]",
    "water": "nwr[natural=water];nwr[waterway]",
    "pool": "nwr[leisure=swimming_pool]",
}
REQUIRED_LAYERS = tuple(LAYER_QUERIES)

LOCAL_DRIVABLE = frozenset(
    {
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "residential",
        "unclassified",
        "service",
        "living_street",
    }
)
ALL_DRIVABLE = LOCAL_DRIVABLE | {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
}
MAPPED_TRANSPORT_SURFACES = ALL_DRIVABLE | {
    "bridleway",
    "construction",
    "cycleway",
    "footway",
    "path",
    "pedestrian",
    "steps",
    "track",
}

# Per-class planting rings, measured from the road centreline, restored from the
# archived study definition (``nature/scripts/osm/osm_rasterize.py``).  ``inner``
# is the carriageway half-width, so it doubles as the road-surface exclusion
# radius; ``outer`` caps the planting strip.  A verge centre therefore always
# sits beside the pavement, never on it.
ROAD_RING_WIDTHS = {
    "motorway": (15.0, 25.0),
    "motorway_link": (15.0, 25.0),
    "trunk": (12.0, 20.0),
    "trunk_link": (12.0, 20.0),
    "primary": (8.0, 14.0),
    "primary_link": (8.0, 14.0),
    "secondary": (6.0, 11.0),
    "secondary_link": (6.0, 11.0),
    "tertiary": (5.0, 9.0),
    "tertiary_link": (5.0, 9.0),
    "residential": (3.5, 7.0),
    "unclassified": (3.0, 6.0),
    "service": (3.0, 6.0),
    "living_street": (3.0, 6.0),
    "pedestrian": (1.0, 3.0),
    "footway": (1.0, 3.0),
    "path": (1.0, 3.0),
    "cycleway": (1.0, 3.0),
    "track": (2.0, 5.0),
    "steps": (1.0, 3.0),
    "bridleway": (1.0, 3.0),
    "construction": (4.0, 8.0),
}
DEFAULT_RING = (4.0, 8.0)
LINEAR_WATER_HALF_WIDTH_M = 2.0


def road_ring(highway: str) -> tuple[float, float]:
    """Return the ``(inner, outer)`` planting ring for one highway class."""
    return ROAD_RING_WIDTHS.get(highway, DEFAULT_RING)


@dataclass(frozen=True)
class OSMPlacementMasks:
    centre_eligible: np.ndarray
    osm_buildings: np.ndarray
    street_analysis: np.ndarray
    layer_masks: dict[str, np.ndarray]
    audit: dict


def _required_geojsons(osm_dir: Path) -> dict[str, Path]:
    paths = {layer: osm_dir / f"{layer}.geojson" for layer in REQUIRED_LAYERS}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "OSM eligibility requires all downloaded layers; missing "
            f"{missing} below {osm_dir}. Run `tree-placement-osm download`."
        )
    return paths


def _feature_collection(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid OSM GeoJSON {path}: {error}") from error
    if payload.get("type") != "FeatureCollection" or not isinstance(
        payload.get("features"), list
    ):
        raise ValueError(f"OSM layer is not a GeoJSON FeatureCollection: {path}")
    return payload["features"]


def _projected_geometries(path: Path, dst_crs) -> list:
    from rasterio.warp import transform_geom
    from shapely.geometry import shape

    geometries = []
    for feature in _feature_collection(path):
        raw = feature.get("geometry")
        if not raw:
            continue
        try:
            geometry = shape(transform_geom("EPSG:4326", dst_crs, raw))
        except Exception as error:
            raise ValueError(f"could not project geometry in {path}: {error}") from error
        if geometry.is_empty:
            continue
        geometries.append((geometry, feature.get("properties") or {}))
    return geometries


def _feature_mask(
    path: Path,
    dst_crs,
    transform,
    shape: tuple[int, int],
    *,
    linear_buffer_m: float | None = None,
) -> np.ndarray:
    from rasterio.features import rasterize

    geometries = []
    for geometry, properties in _projected_geometries(path, dst_crs):
        if geometry.geom_type in {"Polygon", "MultiPolygon"}:
            geometries.append((geometry, properties))
        elif linear_buffer_m is not None and geometry.geom_type in {
            "LineString",
            "MultiLineString",
        }:
            geometries.append((geometry.buffer(linear_buffer_m), properties))
    if not geometries:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        ((geometry, 1) for geometry, _ in geometries),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _road_masks(
    path: Path,
    dst_crs,
    transform,
    shape: tuple[int, int],
    road_scope: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    allowed = LOCAL_DRIVABLE if road_scope == "local" else ALL_DRIVABLE
    centreline_by_class: dict[str, list] = {}
    area_by_class: dict[str, list] = {}
    for geometry, properties in _projected_geometries(path, dst_crs):
        highway = str(properties.get("highway", ""))
        if highway not in MAPPED_TRANSPORT_SURFACES:
            continue
        if geometry.geom_type in {"Polygon", "MultiPolygon"}:
            # An area-mapped surface (a plaza, a service yard) is paved all the
            # way across.  Burn the polygon itself so the interior is excluded,
            # and keep its boundary as the centreline that seeds the ring.
            area_by_class.setdefault(highway, []).append(geometry)
            geometry = geometry.boundary
        if geometry.geom_type not in {"LineString", "MultiLineString"}:
            continue
        centreline_by_class.setdefault(highway, []).append(geometry)

    x_resolution = abs(float(transform.a))
    y_resolution = abs(float(transform.e))
    if not np.isclose(x_resolution, y_resolution):
        raise ValueError("OSM eligibility requires square raster pixels")
    resolution = x_resolution

    def burn(geometries) -> np.ndarray:
        return rasterize(
            ((geometry, 1) for geometry in geometries),
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)

    road_surface = np.zeros(shape, dtype=bool)
    street_carriageway = np.zeros(shape, dtype=bool)
    verge = np.zeros(shape, dtype=bool)
    surface_feature_count = 0
    verge_feature_count = 0
    for highway in set(centreline_by_class) | set(area_by_class):
        inner_m, outer_m = road_ring(highway)
        centrelines = centreline_by_class.get(highway, [])
        areas = area_by_class.get(highway, [])
        surface_feature_count += len(centrelines)

        filled = burn(areas) if areas else np.zeros(shape, dtype=bool)
        class_surface = filled.copy()
        class_ring = np.zeros(shape, dtype=bool)
        if centrelines:
            burned = burn(centrelines)
            if burned.any():
                distance_m = distance_transform_edt(~burned) * resolution
                class_surface |= distance_m <= inner_m
                # The ring starts at the carriageway edge, so a verge centre is
                # beside the pavement by construction.  Area interiors are
                # subtracted for the same reason.
                class_ring = (distance_m > inner_m) & (distance_m <= outer_m)

        road_surface |= class_surface
        if highway not in allowed:
            continue
        verge_feature_count += len(centrelines)
        street_carriageway |= class_surface
        verge |= class_ring & ~filled
    # A ring drawn beside one class can still land on a wider neighbouring road,
    # so subtract every mapped paved surface, not just the hosting class.
    verge &= ~road_surface
    return road_surface, street_carriageway, verge, surface_feature_count, verge_feature_count


def build_osm_placement_masks(
    osm_dir: Path,
    reference_tif: Path,
    placement_domain: str,
    road_scope: str = "all",
) -> OSMPlacementMasks:
    """Build centre-eligibility and supplemental-building masks on a raster grid."""
    import rasterio

    if placement_domain not in {"everywhere", "street-verge"}:
        raise ValueError("placement_domain must be 'everywhere' or 'street-verge'")
    if road_scope not in {"local", "all"}:
        raise ValueError("road_scope must be 'local' or 'all'")
    paths = _required_geojsons(osm_dir)
    with rasterio.open(reference_tif) as source:
        if source.crs is None or not source.crs.is_projected:
            raise ValueError("OSM eligibility requires a projected reference raster")
        dst_crs = source.crs
        transform = source.transform
        shape = source.shape
        if not np.isclose(abs(float(transform.a)), 1.0) or not np.isclose(
            abs(float(transform.e)), 1.0
        ):
            raise ValueError("OSM placement masks require the study's 1 m grid")

    osm_buildings = _feature_mask(paths["building"], dst_crs, transform, shape)
    parking = _feature_mask(paths["parking"], dst_crs, transform, shape)
    residential = _feature_mask(paths["residential"], dst_crs, transform, shape)
    sidewalk = _feature_mask(
        paths["sidewalk"], dst_crs, transform, shape, linear_buffer_m=1.0
    )
    water = _feature_mask(
        paths["water"],
        dst_crs,
        transform,
        shape,
        linear_buffer_m=LINEAR_WATER_HALF_WIDTH_M,
    )
    pools = _feature_mask(paths["pool"], dst_crs, transform, shape)
    (
        roads,
        street_carriageway,
        verge,
        road_surface_feature_count,
        verge_road_feature_count,
    ) = _road_masks(paths["highway"], dst_crs, transform, shape, road_scope)
    common_excluded = osm_buildings | parking | water | pools
    if placement_domain == "street-verge":
        # ``roads`` is the per-class paved surface for *every* mapped highway,
        # not just the verge-hosting classes, so it has to be excluded here as
        # well; otherwise a ring beside a residential street could still sit on
        # an adjacent motorway carriageway.
        street_domain = (verge | sidewalk) & ~roads
        centre_eligible = street_domain & ~common_excluded
        applied_excluded = common_excluded | roads
    else:
        applied_excluded = common_excluded | roads | sidewalk
        centre_eligible = ~applied_excluded

    from ..experiment_contract import sha256_file

    manifest_path = osm_dir / "manifest.json"
    source_files = {
        layer: {
            "path": f"osm/{path.name}",
            "sha256": sha256_file(path),
            "feature_count": len(_feature_collection(path)),
        }
        for layer, path in paths.items()
    }
    audit = {
        "placement_domain": placement_domain,
        "road_scope": road_scope,
        "osm_layers": list(REQUIRED_LAYERS),
        "osm_sources": source_files,
        "osm_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "reference_raster": {
            "path": reference_tif.name,
            "sha256": sha256_file(reference_tif),
            "crs": str(dst_crs),
            "transform": list(transform),
            "shape": list(shape),
        },
        "road_surface_features_used": road_surface_feature_count,
        "verge_road_features_used": verge_road_feature_count,
        "verge_rings_m": (
            {
                highway: list(road_ring(highway))
                for highway in sorted(
                    LOCAL_DRIVABLE if road_scope == "local" else ALL_DRIVABLE
                )
            }
            if placement_domain == "street-verge"
            else None
        ),
        "verge_default_ring_m": (
            list(DEFAULT_RING) if placement_domain == "street-verge" else None
        ),
        "verge_reference": "road-centreline" if placement_domain == "street-verge" else None,
        "sidewalks_supplement_verge": placement_domain == "street-verge",
        "linear_water_half_width_m": LINEAR_WATER_HALF_WIDTH_M,
        "centre_eligible_pixels_before_physical_screen": int(centre_eligible.sum()),
        "applied_exclusion_union_pixels": int(applied_excluded.sum()),
        "applied_exclusion_layers": (
            ["osm_building", "road_surface", "parking", "water", "pool"]
            if placement_domain == "street-verge"
            else ["osm_building", "road_surface", "sidewalk", "parking", "water", "pool"]
        ),
        "residential_layer_is_audit_only": True,
        "layer_pixels": {
            "osm_building": int(osm_buildings.sum()),
            "road_surface": int(roads.sum()),
            "street_carriageway": int(street_carriageway.sum()),
            "parking": int(parking.sum()),
            "residential": int(residential.sum()),
            "sidewalk": int(sidewalk.sum()),
            "water": int(water.sum()),
            "pool": int(pools.sum()),
        },
        "scope_note": (
            "OSM screening does not establish ownership, utility clearance, soil volume, "
            "or planting permission."
        ),
    }
    layer_masks = {
        "building": osm_buildings,
        "road_surface": roads,
        "street_verge": verge,
        "street_carriageway": street_carriageway,
        "sidewalk": sidewalk,
        "parking": parking,
        "residential": residential,
        "water": water,
        "pool": pools,
        "general_eligible": ~(common_excluded | roads | sidewalk),
    }
    street_analysis = (street_carriageway | sidewalk) & ~osm_buildings
    return OSMPlacementMasks(
        centre_eligible,
        osm_buildings,
        street_analysis,
        layer_masks,
        audit,
    )


def _overpass_query(layer_query: str, bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    clauses = []
    for expression in layer_query.split(";"):
        clauses.append(f"{expression}({south},{west},{north},{east});")
    return "[out:json][timeout:120];(" + "".join(clauses) + ");out geom;"


def _request_overpass(query: str, endpoint: str) -> bytes:
    request = urllib.request.Request(
        endpoint,
        data=query.encode("utf-8"),
        headers={"User-Agent": "tree-placement-osm/1.0.0 (research software)"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _relation_geometry(members: list[dict]) -> dict | None:
    """Assemble OSM multipolygon members while retaining inner holes."""
    from shapely.geometry import LineString, mapping
    from shapely.ops import polygonize, unary_union

    rings: dict[str, list] = {"outer": [], "inner": []}
    for member in members:
        points = member.get("geometry")
        if member.get("type") != "way" or not points:
            continue
        coordinates = [
            (point["lon"], point["lat"])
            for point in points
            if "lon" in point and "lat" in point
        ]
        if len(coordinates) < 2:
            continue
        role = "inner" if member.get("role") == "inner" else "outer"
        rings[role].append(LineString(coordinates))
    outer_polygons = list(polygonize(unary_union(rings["outer"])))
    if not outer_polygons:
        return None
    geometry = unary_union(outer_polygons)
    inner_polygons = list(polygonize(unary_union(rings["inner"])))
    if inner_polygons:
        geometry = geometry.difference(unary_union(inner_polygons))
    if geometry.is_empty:
        return None
    return mapping(geometry)


def osm_to_geojson(payload: bytes) -> dict:
    """Convert the ``out geom`` subset used here to a GeoJSON FeatureCollection."""
    data = json.loads(payload)
    features = []
    for element in data.get("elements", []):
        element_type = element.get("type")
        geometry_points = element.get("geometry")
        if element_type == "node":
            if element.get("lon") is None or element.get("lat") is None:
                continue
            geometry = {
                "type": "Point",
                "coordinates": [element["lon"], element["lat"]],
            }
        elif element_type == "way" and geometry_points:
            coordinates = [
                [point["lon"], point["lat"]]
                for point in geometry_points
                if "lon" in point and "lat" in point
            ]
            if len(coordinates) < 2:
                continue
            closed = len(coordinates) >= 4 and coordinates[0] == coordinates[-1]
            geometry = {
                "type": "Polygon" if closed else "LineString",
                "coordinates": [coordinates] if closed else coordinates,
            }
        elif element_type == "relation":
            geometry = _relation_geometry(element.get("members", []))
            if geometry is None:
                continue
        else:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": element.get("tags") or {},
                "id": f"{element_type}/{element.get('id')}",
            }
        )
    return {"type": "FeatureCollection", "features": features}


def download_osm_layers(city_dir: Path, *, force: bool = False) -> dict:
    """Download all required OSM constraint layers for one raster study domain."""
    import rasterio
    from rasterio.warp import transform_bounds

    reference = city_dir / "CDSM.tif"
    if not reference.is_file():
        raise FileNotFoundError(f"reference canopy raster not found: {reference}")
    with rasterio.open(reference) as source:
        if source.crs is None:
            raise ValueError("CDSM.tif has no coordinate system")
        west, south, east, north = transform_bounds(
            source.crs, "EPSG:4326", *source.bounds, densify_pts=21
        )
    bbox = (west, south, east, north)
    osm_dir = city_dir / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "bbox_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "layers": {},
    }

    for layer, expression in LAYER_QUERIES.items():
        output = osm_dir / f"{layer}.geojson"
        if output.is_file() and not force:
            features = _feature_collection(output)
            manifest["layers"][layer] = {
                "status": "cached",
                "feature_count": len(features),
            }
            continue
        query = _overpass_query(expression, bbox)
        last_error = None
        for attempt in range(6):
            endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
            try:
                geojson = osm_to_geojson(_request_overpass(query, endpoint))
                output.write_text(json.dumps(geojson, separators=(",", ":")) + "\n")
                manifest["layers"][layer] = {
                    "status": "downloaded",
                    "feature_count": len(geojson["features"]),
                    "endpoint": endpoint,
                }
                time.sleep(1.5)
                break
            # OSError covers URLError/HTTPError/TimeoutError and the local write;
            # ValueError covers json.JSONDecodeError, which is how a rate-limited
            # or overloaded Overpass mirror fails -- it answers with an HTML or
            # text body, so the decode breaks rather than the transport.
            except (OSError, ValueError) as error:
                last_error = error
                if attempt == 5:
                    raise RuntimeError(
                        f"OSM download failed for {layer} after six attempts: {error}"
                    ) from error
                time.sleep(min(2**attempt, 30))
        if last_error is not None and layer not in manifest["layers"]:
            raise RuntimeError(f"OSM download failed for {layer}: {last_error}")

    (osm_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_mask(path: Path, mask: np.ndarray, reference_tif: Path) -> None:
    import rasterio

    with rasterio.open(reference_tif) as source:
        profile = source.profile.copy()
    # No nodata: 0 is a meaningful value here ("screened out"), and declaring it
    # nodata would make every ineligible pixel vanish from a masked read.
    profile.update(count=1, dtype="uint8", nodata=None, compress="lzw")
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(mask.astype("uint8"), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--city-dir", type=Path, required=True)
    download.add_argument("--force", action="store_true")
    mask = subparsers.add_parser("mask")
    mask.add_argument("--city-dir", type=Path, required=True)
    mask.add_argument(
        "--placement-domain", choices=("everywhere", "street-verge"), required=True
    )
    mask.add_argument("--road-scope", choices=("local", "all"), default="all")
    mask.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "download":
        print(json.dumps(download_osm_layers(args.city_dir, force=args.force), indent=2))
        return
    reference = args.city_dir / "CDSM.tif"
    masks = build_osm_placement_masks(
        args.city_dir / "osm",
        reference,
        args.placement_domain,
        args.road_scope,
    )
    _write_mask(args.output, masks.centre_eligible, reference)
    print(json.dumps(masks.audit, indent=2))


if __name__ == "__main__":
    main()
